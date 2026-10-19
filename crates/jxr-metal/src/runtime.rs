use std::{cell::RefCell, collections::VecDeque, ptr::NonNull, rc::Rc};

use jxr_core::device_plan::OutputParameterWords;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLCompileOptions, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLGPUFamily, MTLLibrary,
};

use crate::{kernels, output_plan::StorePipeline};

pub(crate) type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

// Output parameter words that only select per-format branches. Geometry and
// crop (0-3) and the planar plane index (24) stay dynamic, so one
// specialization serves every image size of a format.
const SPECIALIZED_OUTPUT_WORDS: [usize; 22] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26,
];
// Distinct output formats whose specialized store pipelines stay compiled.
const RETAINED_OUTPUT_SPECIALIZATIONS: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
struct OutputSpecializationKey {
    pipeline: StorePipeline,
    words: [u32; SPECIALIZED_OUTPUT_WORDS.len()],
}

// Four queues outperformed one and eight on the initial 16-core M4 Pro target.
// Exact caller-queue sessions intentionally bypass this default.
const DEFAULT_BATCH_QUEUE_COUNT: usize = 4;

pub(crate) struct MetalRuntime {
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(crate) batch_queues: Vec<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
    pub(crate) buffer_pools: Rc<crate::buffer_pool::MetalBufferPools>,
    pub(crate) upload_cache: Rc<crate::upload_cache::CoefficientUploadCache>,
    pub(crate) overlap_schedules: crate::overlap_plan::OverlapScheduleCache,
    pub(crate) dequant_transform: Pipeline,
    pub(crate) batch_dequant_transform: Pipeline,
    pub(crate) overlap_first: Pipeline,
    pub(crate) hp_transform: Pipeline,
    pub(crate) batch_hp_transform: Pipeline,
    pub(crate) overlap_second: Pipeline,
    pub(crate) overlap_batch: Pipeline,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    output_specializations: RefCell<VecDeque<(OutputSpecializationKey, Pipeline)>>,
}

impl MetalRuntime {
    pub(crate) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: Option<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
    ) -> Result<Self, crate::MetalError> {
        if !device.hasUnifiedMemory() || !device.supportsFamily(MTLGPUFamily::Apple7) {
            return Err(crate::MetalError::UnsupportedDevice {
                reason: "JPEG XR Metal reconstruction requires an M1-or-newer Apple GPU",
            });
        }
        let source = kernels::source();
        let library = precise_library(device, &source)?;
        let queue_was_supplied = queue.is_some();
        let queue = queue.map_or_else(|| j2k_metal_support::checked_command_queue(device), Ok)?;
        let mut batch_queues = vec![queue.clone()];
        if !queue_was_supplied {
            for _ in 1..DEFAULT_BATCH_QUEUE_COUNT {
                batch_queues.push(j2k_metal_support::checked_command_queue(device)?);
            }
        }
        Ok(Self {
            queue,
            batch_queues,
            buffer_pools: Rc::new(crate::buffer_pool::MetalBufferPools::new(device)),
            upload_cache: Rc::new(crate::upload_cache::CoefficientUploadCache::new(device)),
            overlap_schedules: crate::overlap_plan::OverlapScheduleCache::new(),
            dequant_transform: j2k_metal_support::named_pipeline(
                device,
                &library,
                "jxr_dequantize_first_transform",
            )?,
            batch_dequant_transform: j2k_metal_support::named_pipeline(
                device,
                &library,
                "jxr_dequantize_first_transform_batch",
            )?,
            overlap_first: j2k_metal_support::named_pipeline(
                device,
                &library,
                "jxr_first_overlap",
            )?,
            hp_transform: j2k_metal_support::named_pipeline(
                device,
                &library,
                "jxr_highpass_second_transform",
            )?,
            batch_hp_transform: j2k_metal_support::named_pipeline(
                device,
                &library,
                "jxr_highpass_second_transform_batch",
            )?,
            overlap_second: j2k_metal_support::named_pipeline(
                device,
                &library,
                "jxr_second_overlap",
            )?,
            overlap_batch: pipeline(device, &library, "jxr_overlap_batch")?,
            library,
            output_specializations: RefCell::new(VecDeque::new()),
        })
    }

    /// Returns the output store compiled for this exact format.
    ///
    /// Function constants fold the per-format branches of the generic store,
    /// which is otherwise instruction-bound at low GPU clocks. Specializations
    /// compile on first use and are retained per session.
    pub(crate) fn output_pipeline(
        &self,
        kind: StorePipeline,
        params: &OutputParameterWords,
    ) -> Result<Pipeline, crate::MetalError> {
        let key = OutputSpecializationKey {
            pipeline: kind,
            words: SPECIALIZED_OUTPUT_WORDS.map(|index| params[index]),
        };
        if let Some((_, pipeline)) = self
            .output_specializations
            .borrow()
            .iter()
            .find(|(cached, _)| *cached == key)
        {
            return Ok(pipeline.clone());
        }
        let pipeline = specialized_pipeline(
            &self.queue.device(),
            &self.library,
            kind.entrypoint(),
            &key.words,
        )?;
        let mut cache = self.output_specializations.borrow_mut();
        if cache.len() == RETAINED_OUTPUT_SPECIALIZATIONS {
            cache.pop_front();
        }
        cache.push_back((key, pipeline.clone()));
        Ok(pipeline)
    }
}

fn specialized_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
    words: &[u32; SPECIALIZED_OUTPUT_WORDS.len()],
) -> Result<Pipeline, j2k_metal_support::MetalSupportError> {
    let values = MTLFunctionConstantValues::new();
    for (&index, value) in SPECIALIZED_OUTPUT_WORDS.iter().zip(words) {
        // SAFETY: `value` points to a live u32 for the duration of the call,
        // matching the shader's `uint` function constant declared at `index`.
        unsafe {
            values.setConstantValue_type_atIndex(
                NonNull::from(value).cast(),
                MTLDataType::UInt,
                index,
            );
        }
    }
    let function = library
        .newFunctionWithName_constantValues_error(&NSString::from_str(name), &values)
        .map_err(
            |error| j2k_metal_support::MetalSupportError::PipelineFunction {
                function_name: name.to_owned(),
                message: error.localizedDescription().to_string(),
            },
        )?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(
            |error| j2k_metal_support::MetalSupportError::PipelineState {
                function_name: name.to_owned(),
                message: error.localizedDescription().to_string(),
            },
        )
}

fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Pipeline, j2k_metal_support::MetalSupportError> {
    j2k_metal_support::named_pipeline(device, library, name)
}

fn precise_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, j2k_metal_support::MetalSupportError> {
    let options = MTLCompileOptions::new();
    #[allow(
        deprecated,
        reason = "supports the full Apple-silicon deployment range"
    )]
    options.setFastMathEnabled(false);
    let source = NSString::from_str(source);
    device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(
            |error| j2k_metal_support::MetalSupportError::ShaderLibrary {
                message: error.localizedDescription().to_string(),
            },
        )
}

#[cfg(test)]
mod tests {
    use super::MetalRuntime;
    use objc2::rc::Retained;

    #[test]
    fn packages_precise_reconstruction_library() {
        let device = j2k_metal_support::system_default_device().expect("Metal device");
        let runtime = MetalRuntime::build(&device, None).expect("JXR Metal pipeline compilation");
        assert_eq!(runtime.batch_queues.len(), 4);
    }

    #[test]
    fn caller_queue_disables_cross_queue_batch_scheduling() {
        let device = j2k_metal_support::system_default_device().expect("Metal device");
        let queue = j2k_metal_support::checked_command_queue(&device).unwrap();
        let runtime = MetalRuntime::build(&device, Some(queue.clone())).unwrap();
        assert_eq!(runtime.batch_queues.len(), 1);
        assert_eq!(
            Retained::as_ptr(&runtime.batch_queues[0]),
            Retained::as_ptr(&queue)
        );
    }
}
