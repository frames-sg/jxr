// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr_core::{BackendRequest, ChannelLayout, PixelFormat, SurfaceLayout};
use jxr_cuda::{
    CUDA_AUTO_THRESHOLD, CudaDecodePlan, CudaDecoderSession, CudaError, CudaRouteDecision,
    DenseCudaBatchLayout, RECONSTRUCTION_KERNELS, plan_cuda_route,
};

fn output() -> SurfaceLayout {
    SurfaceLayout::tightly_packed(17, 11, PixelFormat::U16(ChannelLayout::Rgba), 1).unwrap()
}

#[test]
fn route_is_strict_for_explicit_cuda_and_falls_back_only_for_auto() {
    assert!(matches!(
        plan_cuda_route(BackendRequest::Cuda, 1, false, false),
        Err(CudaError::Unavailable)
    ));
    assert_eq!(
        plan_cuda_route(BackendRequest::Auto, u64::MAX, false, false).unwrap(),
        CudaRouteDecision::Cpu
    );
    assert_eq!(
        plan_cuda_route(BackendRequest::Auto, CUDA_AUTO_THRESHOLD, true, false).unwrap(),
        CudaRouteDecision::Cuda
    );
}

#[test]
fn metadata_plan_and_dense_batch_sizes_are_checked() {
    assert!(CudaDecodePlan::new(64, 256, output()).is_ok());
    assert!(CudaDecodePlan::new(64, 255, output()).is_err());
    let dense = DenseCudaBatchLayout::new(output(), 128).unwrap();
    assert_eq!(dense.image_count(), 128);
    assert_eq!(dense.image_stride_bytes(), 17 * 11 * 4 * 2);
    assert!(DenseCudaBatchLayout::new(output(), 0).is_err());
    assert!(DenseCudaBatchLayout::new(output(), usize::MAX).is_err());

    let mut abi_overflow = output();
    abi_overflow.byte_len = u32::MAX as usize + 1;
    assert!(CudaDecodePlan::new(64, 256, abi_overflow).is_err());
}

#[test]
fn complete_cuda_kernel_manifest_is_packaged() {
    let names = RECONSTRUCTION_KERNELS
        .iter()
        .map(|kernel| kernel.entrypoint)
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "jxr_dequantize_first_transform",
            "jxr_first_overlap",
            "jxr_highpass_second_transform",
            "jxr_second_overlap",
            "jxr_output_u8",
        ]
    );
}

#[test]
fn availability_probe_does_not_require_a_linked_cuda_toolkit() {
    let available = CudaDecoderSession::is_available();
    if !available {
        assert!(matches!(
            CudaDecoderSession::system_default(),
            Err(CudaError::Unavailable)
        ));
    }
}
