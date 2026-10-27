// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    static_mut_refs,
    reason = "CUDA shared arrays are scoped to a thread block"
)]

use cuda_device::{
    SharedArray,
    atomic::{AtomicOrdering, DeviceAtomicU32},
    kernel, thread,
};
use cuda_host::cuda_module;
use jxr_math::tables::reconstruction_abi::*;
use jxr_math::{
    MathError,
    arithmetic::{checked_add, checked_i64_to_i32, checked_mul, checked_sub},
};

// CUDA Oxide cannot lower the nested ControlFlow layout produced by `?`.
macro_rules! math_try {
    ($value:expr) => {
        match $value {
            Ok(value) => value,
            Err(error) => return Err(error),
        }
    };
}

mod output;
mod reconstruction;

#[inline(always)]
fn device_load<T: Copy>(pointer: *const T, index: u32) -> T {
    // SAFETY: Host plans validate all buffer ranges and device descriptor offsets.
    unsafe { *pointer.add(index as usize) }
}

#[inline(always)]
fn store<T>(pointer: *mut T, index: u32, value: T) {
    // SAFETY: Host plans validate destination extents; each work item owns its writes.
    unsafe {
        *pointer.add(index as usize) = value;
    }
}

#[inline(always)]
fn dequantize(value: i32, quantizer: u32) -> Result<i32, MathError> {
    checked_i64_to_i32(value as i64 * quantizer as i64)
}

fn fail(status: *mut u32, code: u32) {
    // SAFETY: Every submission retains one aligned status word until completion.
    let atomic = unsafe { DeviceAtomicU32::from_ptr(status) };
    let _ = atomic.compare_exchange(0, code, AtomicOrdering::Relaxed, AtomicOrdering::Relaxed);
}

fn block_failed(status: *mut u32) -> bool {
    static mut STATUS: SharedArray<u32, 1> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x()
        + thread::blockDim_x()
            * (thread::threadIdx_y() + thread::blockDim_y() * thread::threadIdx_z());
    // SAFETY: Only thread zero writes; every thread reaches the barrier before reading.
    unsafe {
        if tid == 0 {
            STATUS[0] = DeviceAtomicU32::from_ptr(status).load(AtomicOrdering::Relaxed);
        }
        thread::sync_threads();
        STATUS[0] != 0
    }
}

fn output_store(
    samples: *const i32,
    planes: *const JxrSamplePlaneAbi,
    surfaces: *const JxrSurfacePlaneAbi,
    output: *mut u8,
    status: *mut u32,
    params: JxrOutputAbi,
    output_base: u32,
    kind: u32,
) {
    let x = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    let y = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
    let surface = device_load(surfaces, params.output_plane);
    let failed = block_failed(status);
    let width = if kind == 0 {
        (surface.width + 7) / 8
    } else {
        surface.width
    };
    if x >= width || y >= surface.height || failed {
        return;
    }
    // SAFETY: Output allocation and the per-image base are validated before launch.
    let output = unsafe { output.add(output_base as usize) };
    if output::write(samples, planes, surface, output, params, x, y, kind).is_err() {
        fail(status, 16);
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn jxr_dequantize_first_transform(
        packed: *const i32,
        macroblocks: *const JxrMacroblockAbi,
        low_plane: *mut i32,
        status: *mut u32,
        plane: JxrPlaneAbi,
    ) {
        let gid = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let failed = block_failed(status);
        if gid >= plane.macroblock_count || failed {
            return;
        }
        if reconstruction::first_transform(packed, macroblocks, low_plane, plane, gid).is_err() {
            fail(status, 1);
        }
    }

    #[kernel]
    pub unsafe fn jxr_highpass_second_transform(
        packed: *const i32,
        macroblocks: *const JxrMacroblockAbi,
        low_plane: *const i32,
        samples: *mut i32,
        status: *mut u32,
        plane: JxrPlaneAbi,
    ) {
        static mut HIGH: SharedArray<i32, 256> = SharedArray::UNINIT;
        let group = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let threads = thread::blockDim_x();
        let failed = block_failed(status);
        if group >= plane.macroblock_count || failed {
            return;
        }
        let metadata = device_load(macroblocks, plane.macroblock_offset + group);
        let count = plane.block_columns * plane.block_rows;
        // SAFETY: Each lane initializes distinct entries, all lanes synchronize before prediction.
        let high = unsafe { HIGH.as_mut_ptr() };
        let mut index = tid;
        while index < count * 16 {
            let value = if metadata.bands < 2 || index & 15 == 0 {
                0
            } else {
                device_load(packed, metadata.coefficient_offset + index)
            };
            store(high, index, value);
            index += threads;
        }
        thread::sync_threads();
        if tid == 0 && reconstruction::predict_high(high, metadata, plane).is_err() {
            fail(status, 3);
        }
        thread::sync_threads();
        if block_failed(status) {
            return;
        }
        let mut block = tid;
        while block < count {
            if reconstruction::second_transform(high, low_plane, samples, metadata, plane, block)
                .is_err()
            {
                fail(status, 4);
                return;
            }
            block += threads;
        }
    }

    #[kernel]
    pub unsafe fn jxr_first_overlap(
        samples: *mut i32,
        work: *const JxrOverlapWorkAbi,
        status: *mut u32,
        count: u32,
    ) {
        let gid = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let failed = block_failed(status);
        if gid < count
            && !failed
            && reconstruction::overlap(samples, device_load(work, gid)).is_err()
        {
            fail(status, 2);
        }
    }

    #[kernel]
    pub unsafe fn jxr_second_overlap(
        samples: *mut i32,
        work: *const JxrOverlapWorkAbi,
        status: *mut u32,
        count: u32,
    ) {
        let gid = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let failed = block_failed(status);
        if gid < count
            && !failed
            && reconstruction::overlap(samples, device_load(work, gid)).is_err()
        {
            fail(status, 2);
        }
    }

    #[kernel]
    pub unsafe fn jxr_output_bits(
        samples: *const i32,
        planes: *const JxrSamplePlaneAbi,
        surfaces: *const JxrSurfacePlaneAbi,
        output: *mut u8,
        status: *mut u32,
        params: JxrOutputAbi,
        base: u32,
    ) {
        output_store(samples, planes, surfaces, output, status, params, base, 0);
    }
    #[kernel]
    pub unsafe fn jxr_output_u8(
        samples: *const i32,
        planes: *const JxrSamplePlaneAbi,
        surfaces: *const JxrSurfacePlaneAbi,
        output: *mut u8,
        status: *mut u32,
        params: JxrOutputAbi,
        base: u32,
    ) {
        output_store(samples, planes, surfaces, output, status, params, base, 1);
    }
    #[kernel]
    pub unsafe fn jxr_output_u16(
        samples: *const i32,
        planes: *const JxrSamplePlaneAbi,
        surfaces: *const JxrSurfacePlaneAbi,
        output: *mut u8,
        status: *mut u32,
        params: JxrOutputAbi,
        base: u32,
    ) {
        output_store(samples, planes, surfaces, output, status, params, base, 2);
    }
    #[kernel]
    pub unsafe fn jxr_output_i16(
        samples: *const i32,
        planes: *const JxrSamplePlaneAbi,
        surfaces: *const JxrSurfacePlaneAbi,
        output: *mut u8,
        status: *mut u32,
        params: JxrOutputAbi,
        base: u32,
    ) {
        output_store(samples, planes, surfaces, output, status, params, base, 3);
    }
    #[kernel]
    pub unsafe fn jxr_output_i32(
        samples: *const i32,
        planes: *const JxrSamplePlaneAbi,
        surfaces: *const JxrSurfacePlaneAbi,
        output: *mut u8,
        status: *mut u32,
        params: JxrOutputAbi,
        base: u32,
    ) {
        output_store(samples, planes, surfaces, output, status, params, base, 4);
    }
    #[kernel]
    pub unsafe fn jxr_output_f16(
        samples: *const i32,
        planes: *const JxrSamplePlaneAbi,
        surfaces: *const JxrSurfacePlaneAbi,
        output: *mut u8,
        status: *mut u32,
        params: JxrOutputAbi,
        base: u32,
    ) {
        output_store(samples, planes, surfaces, output, status, params, base, 5);
    }
    #[kernel]
    pub unsafe fn jxr_output_f32(
        samples: *const i32,
        planes: *const JxrSamplePlaneAbi,
        surfaces: *const JxrSurfacePlaneAbi,
        output: *mut u8,
        status: *mut u32,
        params: JxrOutputAbi,
        base: u32,
    ) {
        output_store(samples, planes, surfaces, output, status, params, base, 6);
    }
    #[kernel]
    pub unsafe fn jxr_output_packed16(
        samples: *const i32,
        planes: *const JxrSamplePlaneAbi,
        surfaces: *const JxrSurfacePlaneAbi,
        output: *mut u8,
        status: *mut u32,
        params: JxrOutputAbi,
        base: u32,
    ) {
        output_store(samples, planes, surfaces, output, status, params, base, 7);
    }
    #[kernel]
    pub unsafe fn jxr_output_packed32(
        samples: *const i32,
        planes: *const JxrSamplePlaneAbi,
        surfaces: *const JxrSurfacePlaneAbi,
        output: *mut u8,
        status: *mut u32,
        params: JxrOutputAbi,
        base: u32,
    ) {
        output_store(samples, planes, surfaces, output, status, params, base, 8);
    }
}

fn main() {}
