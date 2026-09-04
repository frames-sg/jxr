// SPDX-License-Identifier: MIT OR Apache-2.0

/// Ordered JPEG XR reconstruction phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelStage {
    /// Dequantization and first inverse core transform.
    DequantizeAndFirstTransform,
    /// First overlap pass for overlap mode two.
    FirstOverlap,
    /// HP prediction/dequantization and second inverse transform.
    HighpassAndSecondTransform,
    /// Second overlap pass for overlap modes one and two.
    SecondOverlap,
    /// Chroma/color/alpha reconstruction, clipping, crop, and packing.
    OutputStore,
}

/// Static identity for one Metal phase implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalKernelManifest {
    /// Reconstruction phase.
    pub stage: KernelStage,
    /// Packaged Metal entry point.
    pub entrypoint: &'static str,
}

/// Ordered packaged reconstruction manifest.
pub const RECONSTRUCTION_KERNELS: [MetalKernelManifest; 5] = [
    MetalKernelManifest {
        stage: KernelStage::DequantizeAndFirstTransform,
        entrypoint: "jxr_dequantize_first_transform",
    },
    MetalKernelManifest {
        stage: KernelStage::FirstOverlap,
        entrypoint: "jxr_first_overlap",
    },
    MetalKernelManifest {
        stage: KernelStage::HighpassAndSecondTransform,
        entrypoint: "jxr_highpass_second_transform",
    },
    MetalKernelManifest {
        stage: KernelStage::SecondOverlap,
        entrypoint: "jxr_second_overlap",
    },
    MetalKernelManifest {
        stage: KernelStage::OutputStore,
        entrypoint: "jxr_output_u8",
    },
];

/// Assemble one Metal library from the shared arithmetic and focused phase modules.
#[cfg(target_os = "macos")]
pub(crate) fn source() -> String {
    [
        jxr_math::tables::METAL_RECONSTRUCTION_CONSTANTS,
        common::SOURCE,
        include_str!("first_transform.metal"),
        include_str!("overlap.metal"),
        include_str!("highpass_transform.metal"),
        include_str!("output.metal"),
    ]
    .join("\n")
}
#[cfg(target_os = "macos")]
mod common;
