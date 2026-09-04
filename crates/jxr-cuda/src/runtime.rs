// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::HashMap,
    panic::AssertUnwindSafe,
    sync::{Arc, Mutex},
};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaStream},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use crate::{CudaError, buffer_pool::CudaBufferPool, kernels, upload_cache::CudaUploadCache};

pub(crate) const BATCH_SCRATCH_BUDGET: usize = 256 * 1024 * 1024;
const DEFAULT_STREAMS: usize = 4;

const ENTRYPOINTS: [&str; 13] = [
    "jxr_dequantize_first_transform",
    "jxr_first_overlap",
    "jxr_highpass_second_transform",
    "jxr_second_overlap",
    "jxr_output_bits",
    "jxr_output_u8",
    "jxr_output_u16",
    "jxr_output_i16",
    "jxr_output_i32",
    "jxr_output_f16",
    "jxr_output_f32",
    "jxr_output_packed16",
    "jxr_output_packed32",
];

pub(crate) struct CudaRuntime {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) streams: Arc<[Arc<CudaStream>]>,
    _module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
    pub(crate) buffer_pool: Arc<CudaBufferPool>,
    pub(crate) upload_cache: Arc<CudaUploadCache>,
    pub(crate) submission_lock: Mutex<()>,
}

impl core::fmt::Debug for CudaRuntime {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaRuntime")
            .field("device_ordinal", &self.context.ordinal())
            .field("stream_count", &self.streams.len())
            .finish_non_exhaustive()
    }
}

impl CudaRuntime {
    pub(crate) fn build(device_ordinal: usize) -> Result<Arc<Self>, CudaError> {
        catch_runtime_initialization(|| Self::build_inner(device_ordinal))
    }

    fn build_inner(device_ordinal: usize) -> Result<Arc<Self>, CudaError> {
        let context = CudaContext::new(device_ordinal)?;
        let (major, minor) = context.compute_capability()?;
        if major < 5 {
            return Err(CudaError::UnsupportedDevice {
                reason: "JPEG XR CUDA reconstruction requires compute capability 5.0 or newer",
            });
        }
        let mut options = CompileOptions {
            fmad: Some(false),
            name: Some("jxr_reconstruction.cu".to_owned()),
            ..CompileOptions::default()
        };
        options.options.push("--std=c++17".to_owned());
        options
            .options
            .push(format!("--gpu-architecture=compute_{major}{minor}"));
        let compiled = compile_ptx_with_opts(kernels::source(), options).map_err(|error| {
            CudaError::RuntimeInitialization {
                message: error.to_string(),
            }
        })?;
        let module =
            context
                .load_module(compiled)
                .map_err(|error| CudaError::RuntimeInitialization {
                    message: error.to_string(),
                })?;
        let mut functions = HashMap::with_capacity(ENTRYPOINTS.len());
        for entrypoint in ENTRYPOINTS {
            let function = module.load_function(entrypoint).map_err(|error| {
                CudaError::RuntimeInitialization {
                    message: format!("load CUDA entry point {entrypoint}: {error}"),
                }
            })?;
            functions.insert(entrypoint, function);
        }
        let mut streams = Vec::with_capacity(DEFAULT_STREAMS);
        for _ in 0..DEFAULT_STREAMS {
            streams.push(context.new_stream()?);
        }
        Ok(Arc::new(Self {
            context,
            streams: streams.into(),
            _module: module,
            functions,
            buffer_pool: Arc::new(CudaBufferPool::new()),
            upload_cache: Arc::new(CudaUploadCache::new()),
            submission_lock: Mutex::new(()),
        }))
    }

    pub(crate) fn function(&self, name: &'static str) -> Result<&CudaFunction, CudaError> {
        self.functions.get(name).ok_or(CudaError::StateInvariant {
            state: "CUDA pipeline table",
            reason: "required reconstruction entry point is absent",
        })
    }

    pub(crate) fn stream(&self, index: usize) -> &Arc<CudaStream> {
        &self.streams[index % self.streams.len()]
    }
}

fn catch_runtime_initialization<T>(
    initialize: impl FnOnce() -> Result<T, CudaError>,
) -> Result<T, CudaError> {
    std::panic::catch_unwind(AssertUnwindSafe(initialize)).map_err(|panic| {
        CudaError::RuntimeInitialization {
            message: panic_message(&panic),
        }
    })?
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = panic.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else {
        "CUDA runtime loader panicked without a string diagnostic".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::catch_runtime_initialization;
    use crate::CudaError;

    #[test]
    fn dynamic_loader_panics_become_explicit_initialization_errors() {
        let error = catch_runtime_initialization(|| -> Result<(), CudaError> {
            panic!("missing required CUDA symbol")
        })
        .unwrap_err();
        assert!(matches!(
            error,
            CudaError::RuntimeInitialization { message }
                if message.contains("missing required CUDA symbol")
        ));
    }
}
