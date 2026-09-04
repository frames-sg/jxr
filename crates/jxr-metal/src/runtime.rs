use std::rc::Rc;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLGPUFamily,
    MTLLibrary,
};

use crate::kernels;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

// Four queues outperformed one and eight on the initial 16-core M4 Pro target.
// Exact caller-queue sessions intentionally bypass this default.
const DEFAULT_BATCH_QUEUE_COUNT: usize = 4;

pub(crate) struct MetalRuntime {
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(crate) batch_queues: Vec<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
    pub(crate) buffer_pools: Rc<crate::buffer_pool::MetalBufferPools>,
    pub(crate) upload_cache: Rc<crate::upload_cache::CoefficientUploadCache>,
    pub(crate) dequant_transform: Pipeline,
    pub(crate) batch_dequant_transform: Pipeline,
    pub(crate) overlap_first: Pipeline,
    pub(crate) hp_transform: Pipeline,
    pub(crate) batch_hp_transform: Pipeline,
    pub(crate) overlap_second: Pipeline,
    pub(crate) output_store: OutputPipelines,
}

pub(crate) struct OutputPipelines {
    pub(crate) bits: Pipeline,
    pub(crate) u8: Pipeline,
    pub(crate) u16: Pipeline,
    pub(crate) i16: Pipeline,
    pub(crate) i32: Pipeline,
    pub(crate) f16: Pipeline,
    pub(crate) f32: Pipeline,
    pub(crate) packed16: Pipeline,
    pub(crate) packed32: Pipeline,
}

impl OutputPipelines {
    pub(crate) fn select(
        &self,
        kind: crate::output_plan::StorePipeline,
    ) -> &ProtocolObject<dyn MTLComputePipelineState> {
        match kind {
            crate::output_plan::StorePipeline::Bits => &self.bits,
            crate::output_plan::StorePipeline::U8 => &self.u8,
            crate::output_plan::StorePipeline::U16 => &self.u16,
            crate::output_plan::StorePipeline::I16 => &self.i16,
            crate::output_plan::StorePipeline::I32 => &self.i32,
            crate::output_plan::StorePipeline::F16 => &self.f16,
            crate::output_plan::StorePipeline::F32 => &self.f32,
            crate::output_plan::StorePipeline::Packed16 => &self.packed16,
            crate::output_plan::StorePipeline::Packed32 => &self.packed32,
        }
    }
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
            output_store: OutputPipelines {
                bits: pipeline(device, &library, "jxr_output_bits")?,
                u8: pipeline(device, &library, "jxr_output_u8")?,
                u16: pipeline(device, &library, "jxr_output_u16")?,
                i16: pipeline(device, &library, "jxr_output_i16")?,
                i32: pipeline(device, &library, "jxr_output_i32")?,
                f16: pipeline(device, &library, "jxr_output_f16")?,
                f32: pipeline(device, &library, "jxr_output_f32")?,
                packed16: pipeline(device, &library, "jxr_output_packed16")?,
                packed32: pipeline(device, &library, "jxr_output_packed32")?,
            },
        })
    }
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
