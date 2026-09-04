// SPDX-License-Identifier: MIT OR Apache-2.0

/// Ordered JPEG XR CUDA reconstruction phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelStage {
    /// Dequantization and first inverse core transform.
    DequantizeAndFirstTransform,
    /// First overlap pass for overlap mode two.
    FirstOverlap,
    /// High-pass prediction/dequantization and second inverse transform.
    HighpassAndSecondTransform,
    /// Second overlap pass for overlap modes one and two.
    SecondOverlap,
    /// Chroma/color/alpha reconstruction, crop, clipping, and packing.
    OutputStore,
}

/// Static identity for one CUDA phase implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaKernelManifest {
    /// Reconstruction phase.
    pub stage: KernelStage,
    /// Packaged CUDA C entry point.
    pub entrypoint: &'static str,
}

/// Ordered complete reconstruction manifest.
pub const RECONSTRUCTION_KERNELS: [CudaKernelManifest; 5] = [
    CudaKernelManifest {
        stage: KernelStage::DequantizeAndFirstTransform,
        entrypoint: "jxr_dequantize_first_transform",
    },
    CudaKernelManifest {
        stage: KernelStage::FirstOverlap,
        entrypoint: "jxr_first_overlap",
    },
    CudaKernelManifest {
        stage: KernelStage::HighpassAndSecondTransform,
        entrypoint: "jxr_highpass_second_transform",
    },
    CudaKernelManifest {
        stage: KernelStage::SecondOverlap,
        entrypoint: "jxr_second_overlap",
    },
    CudaKernelManifest {
        stage: KernelStage::OutputStore,
        entrypoint: "jxr_output_u8",
    },
];

pub(crate) fn source() -> String {
    [
        jxr_math::tables::CUDA_RECONSTRUCTION_CONSTANTS,
        include_str!("kernels/common.cu"),
        include_str!("kernels/first_transform.cu"),
        include_str!("kernels/overlap.cu"),
        include_str!("kernels/highpass_transform.cu"),
        include_str!("kernels/output.cu"),
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::source;

    #[test]
    fn packages_every_runtime_entrypoint_without_metal_syntax() {
        let source = source();
        for entrypoint in [
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
        ] {
            assert!(source.contains(entrypoint), "missing {entrypoint}");
        }
        assert!(!source.contains("[[buffer("));
        assert!(!source.contains("metal_stdlib"));
    }

    #[test]
    fn kernel_entry_status_is_broadcast_once_per_thread_block() {
        let source = source();
        assert!(source.contains("bool jxr_block_failed(uint *status)"));
        assert!(source.contains("__shared__ uint jxr_block_status"));
    }
}
