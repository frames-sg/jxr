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
    /// Packaged CUDA Oxide entry point.
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

pub(crate) const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/jxr_reconstruction.ptx"));
