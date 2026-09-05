use jxr_core::device_plan::{OUTPUT_PLANE, SURFACE_HEIGHT, SURFACE_WIDTH};
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::{
    CudaDecodePlan, CudaError,
    abi::{JxrOverlapWorkAbi, JxrPlaneAbi, JxrSamplePlaneAbi, JxrSurfacePlaneAbi},
    output_plan::{OutputDispatchPlan, StorePipeline, build_output_dispatch},
    overlap_plan::{OverlapSchedule, first_overlap_schedule, second_overlap_schedule},
    plan::{CudaReconstructionInput, reconstruction_arena},
    runtime::{BATCH_SCRATCH_BUDGET, CudaRuntime},
    upload_cache::DeviceArena,
};

pub(crate) struct EncodedCudaSubmission {
    pub(crate) completion: CudaEvent,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) output: Option<CudaSlice<u8>>,
    pub(crate) status: CudaSlice<u32>,
    pub(crate) scratch: Vec<CudaSlice<i32>>,
    // These allocations are read asynchronously and must outlive the event.
    pub(crate) _overlap_work: Vec<CudaSlice<JxrOverlapWorkAbi>>,
    pub(crate) _sample_planes: CudaSlice<JxrSamplePlaneAbi>,
    pub(crate) _surface_planes: CudaSlice<JxrSurfacePlaneAbi>,
    pub(crate) _uploads: Vec<Arc<DeviceArena>>,
    pub(crate) runtime: Arc<CudaRuntime>,
    pub(crate) layout: jxr_core::SurfaceLayout,
}

pub(crate) fn encode_owned(
    runtime: &Arc<CudaRuntime>,
    plan: &CudaDecodePlan,
    stream_index: usize,
) -> Result<EncodedCudaSubmission, CudaError> {
    let stream = runtime.stream(stream_index);
    let mut output = stream.alloc_zeros(plan.output().byte_len)?;
    let mut encoded = encode_impl(runtime, plan, &mut output, 0, stream_index)?;
    encoded.output = Some(output);
    Ok(encoded)
}

pub(crate) fn encode_into(
    runtime: &Arc<CudaRuntime>,
    plan: &CudaDecodePlan,
    output: &mut CudaSlice<u8>,
    output_base: usize,
    stream_index: usize,
) -> Result<EncodedCudaSubmission, CudaError> {
    let end =
        output_base
            .checked_add(plan.output().byte_len)
            .ok_or(CudaError::InvalidDestination {
                reason: "CUDA output range overflows usize",
            })?;
    if end > output.len() {
        return Err(CudaError::InvalidDestination {
            reason: "CUDA output range exceeds the destination",
        });
    }
    encode_impl(runtime, plan, output, output_base, stream_index)
}

fn encode_impl(
    runtime: &Arc<CudaRuntime>,
    plan: &CudaDecodePlan,
    output: &mut CudaSlice<u8>,
    output_base: usize,
    stream_index: usize,
) -> Result<EncodedCudaSubmission, CudaError> {
    if plan.scratch_bytes()? > BATCH_SCRATCH_BUDGET {
        return Err(CudaError::ResourceLimit {
            reason: "one image exceeds the CUDA scratch budget",
            requested: plan.scratch_bytes()?,
            maximum: BATCH_SCRATCH_BUDGET,
        });
    }
    let output_base = u32::try_from(output_base).map_err(|_| CudaError::InvalidDestination {
        reason: "CUDA output base exceeds the reconstruction ABI",
    })?;
    let _submission = runtime
        .submission_lock
        .lock()
        .map_err(|_| CudaError::StatePoisoned {
            state: "CUDA submission encoder",
        })?;
    let input = plan.reconstruction()?;
    let stream = runtime.stream(stream_index).clone();
    let mut uploads = Vec::with_capacity(input.arenas.len());
    for arena in input.arenas.iter() {
        uploads.push(runtime.upload_cache.get_or_upload(
            stream_index % runtime.streams.len(),
            &stream,
            arena.source.clone(),
        )?);
    }
    let mut low = runtime.buffer_pool.take(&stream, input.low_len)?;
    let mut samples = runtime.buffer_pool.take(&stream, input.sample_len)?;
    let mut status = stream.alloc_zeros::<u32>(1)?;
    let mut overlap_work = Vec::new();

    launch_first_transforms(runtime, &stream, input, &uploads, &mut low, &mut status)?;
    launch_first_overlaps(
        runtime,
        &stream,
        input,
        &mut low,
        &mut status,
        &mut overlap_work,
    )?;
    launch_second_transforms(
        runtime,
        &stream,
        input,
        &uploads,
        &low,
        &mut samples,
        &mut status,
    )?;
    launch_second_overlaps(
        runtime,
        &stream,
        input,
        &mut samples,
        &mut status,
        &mut overlap_work,
    )?;

    let output_plan = build_output_dispatch(plan)?;
    let sample_planes = stream.clone_htod(&output_plan.samples)?;
    let surface_planes = stream.clone_htod(&output_plan.surfaces)?;
    launch_output(
        runtime,
        &stream,
        &samples,
        &sample_planes,
        &surface_planes,
        output,
        &mut status,
        &output_plan,
        output_base,
    )?;
    let completion = stream.record_event(None)?;
    Ok(EncodedCudaSubmission {
        completion,
        stream,
        output: None,
        status,
        scratch: vec![low, samples],
        _overlap_work: overlap_work,
        _sample_planes: sample_planes,
        _surface_planes: surface_planes,
        _uploads: uploads,
        runtime: runtime.clone(),
        layout: plan.output().clone(),
    })
}

fn launch_first_transforms(
    runtime: &CudaRuntime,
    stream: &Arc<CudaStream>,
    input: &CudaReconstructionInput,
    uploads: &[Arc<DeviceArena>],
    low: &mut CudaSlice<i32>,
    status: &mut CudaSlice<u32>,
) -> Result<(), CudaError> {
    for &plane in input.planes.iter() {
        let abi = JxrPlaneAbi::from_plan(plane)?;
        let arena = arena(uploads, plane.arena_index)?;
        let function = runtime.function("jxr_dequantize_first_transform")?;
        let mut launch = stream.launch_builder(function);
        launch
            .arg(&arena.coefficients)
            .arg(&arena.macroblocks)
            .arg(&mut *low)
            .arg(&mut *status)
            .arg(&abi);
        // SAFETY: The entry point and argument ABI are fixed together in this
        // crate, slices cover validated ranges, and the grid is plan-bounded.
        unsafe {
            launch.launch(linear_config(abi.macroblock_count, 128)?)?;
        }
    }
    Ok(())
}

fn launch_first_overlaps(
    runtime: &CudaRuntime,
    stream: &Arc<CudaStream>,
    input: &CudaReconstructionInput,
    low: &mut CudaSlice<i32>,
    status: &mut CudaSlice<u32>,
    retained: &mut Vec<CudaSlice<JxrOverlapWorkAbi>>,
) -> Result<(), CudaError> {
    for &plane in input.planes.iter() {
        let arena = reconstruction_arena(input, plane.arena_index)?;
        if arena.overlap != jxr_core::OverlapMode::Two {
            continue;
        }
        let schedule = first_overlap_schedule(
            plane,
            arena.hard_tiles,
            &arena.tile_column_widths,
            &arena.tile_row_heights,
        )?;
        launch_schedule(
            runtime,
            stream,
            "jxr_first_overlap",
            low,
            status,
            schedule,
            retained,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_second_transforms(
    runtime: &CudaRuntime,
    stream: &Arc<CudaStream>,
    input: &CudaReconstructionInput,
    uploads: &[Arc<DeviceArena>],
    low: &CudaSlice<i32>,
    samples: &mut CudaSlice<i32>,
    status: &mut CudaSlice<u32>,
) -> Result<(), CudaError> {
    for &plane in input.planes.iter() {
        let abi = JxrPlaneAbi::from_plan(plane)?;
        let arena = arena(uploads, plane.arena_index)?;
        let function = runtime.function("jxr_highpass_second_transform")?;
        let mut launch = stream.launch_builder(function);
        launch
            .arg(&arena.coefficients)
            .arg(&arena.macroblocks)
            .arg(low)
            .arg(&mut *samples)
            .arg(&mut *status)
            .arg(&abi);
        // SAFETY: One block owns one macroblock. Sixteen threads cover at most
        // sixteen 4x4 blocks; shared storage holds the maximum 256 coefficients.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (abi.macroblock_count, 1, 1),
                block_dim: (16, 1, 1),
                shared_mem_bytes: 256 * 4,
            })?;
        }
    }
    Ok(())
}

fn launch_second_overlaps(
    runtime: &CudaRuntime,
    stream: &Arc<CudaStream>,
    input: &CudaReconstructionInput,
    samples: &mut CudaSlice<i32>,
    status: &mut CudaSlice<u32>,
    retained: &mut Vec<CudaSlice<JxrOverlapWorkAbi>>,
) -> Result<(), CudaError> {
    for &plane in input.planes.iter() {
        let arena = reconstruction_arena(input, plane.arena_index)?;
        if arena.overlap == jxr_core::OverlapMode::None {
            continue;
        }
        let schedule = second_overlap_schedule(
            plane,
            arena.hard_tiles,
            &arena.tile_column_widths,
            &arena.tile_row_heights,
        )?;
        launch_schedule(
            runtime,
            stream,
            "jxr_second_overlap",
            samples,
            status,
            schedule,
            retained,
        )?;
    }
    Ok(())
}

fn arena(uploads: &[Arc<DeviceArena>], index: u32) -> Result<&DeviceArena, CudaError> {
    uploads
        .get(usize::try_from(index).map_err(|_| CudaError::InvalidPlan {
            reason: "coefficient arena index does not fit usize",
        })?)
        .map(AsRef::as_ref)
        .ok_or(CudaError::InvalidPlan {
            reason: "coefficient arena index is out of range",
        })
}

#[allow(clippy::too_many_arguments)]
fn launch_output(
    runtime: &CudaRuntime,
    stream: &Arc<CudaStream>,
    samples: &CudaSlice<i32>,
    sample_planes: &CudaSlice<JxrSamplePlaneAbi>,
    surface_planes: &CudaSlice<JxrSurfacePlaneAbi>,
    output: &mut CudaSlice<u8>,
    status: &mut CudaSlice<u32>,
    dispatch: &OutputDispatchPlan,
    output_base: u32,
) -> Result<(), CudaError> {
    let function = runtime.function(dispatch.pipeline.entrypoint())?;
    let plane_count = if dispatch.planar {
        dispatch.surfaces.len()
    } else {
        1
    };
    for plane_index in 0..plane_count {
        let mut params = dispatch.params;
        params[OUTPUT_PLANE] = u32::try_from(plane_index).map_err(|_| CudaError::InvalidPlan {
            reason: "output plane index exceeds the CUDA ABI",
        })?;
        let surface = dispatch.surfaces[plane_index];
        let width = if dispatch.pipeline == StorePipeline::Bits {
            surface[SURFACE_WIDTH].div_ceil(8)
        } else {
            surface[SURFACE_WIDTH]
        };
        let mut launch = stream.launch_builder(function);
        launch
            .arg(samples)
            .arg(sample_planes)
            .arg(surface_planes)
            .arg(&mut *output)
            .arg(&mut *status)
            .arg(&params)
            .arg(&output_base);
        // SAFETY: The output entry point is selected from the validated storage
        // kind, descriptors cover the output allocation, and the two-dimensional
        // launch is clipped again by the kernel before any read or write.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (width.div_ceil(16), surface[SURFACE_HEIGHT].div_ceil(16), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            })?;
        }
    }
    Ok(())
}

fn launch_schedule(
    runtime: &CudaRuntime,
    stream: &Arc<CudaStream>,
    entrypoint: &'static str,
    samples: &mut CudaSlice<i32>,
    status: &mut CudaSlice<u32>,
    schedule: OverlapSchedule,
    retained: &mut Vec<CudaSlice<JxrOverlapWorkAbi>>,
) -> Result<(), CudaError> {
    for work in [schedule.prefix, schedule.filters, schedule.suffix] {
        if work.is_empty() {
            continue;
        }
        let work_count = u32::try_from(work.len()).map_err(|_| CudaError::InvalidPlan {
            reason: "overlap work count exceeds the CUDA ABI",
        })?;
        let work = stream.clone_htod(&work)?;
        let function = runtime.function(entrypoint)?;
        let mut launch = stream.launch_builder(function);
        launch
            .arg(&mut *samples)
            .arg(&work)
            .arg(&mut *status)
            .arg(&work_count);
        // SAFETY: Work descriptors are generated from non-overlapping schedule
        // phases and contain indices checked against the validated plane extents.
        unsafe {
            launch.launch(linear_config(work_count, 128)?)?;
        }
        retained.push(work);
    }
    Ok(())
}

fn linear_config(elements: u32, threads: u32) -> Result<LaunchConfig, CudaError> {
    if elements == 0 || threads == 0 {
        return Err(CudaError::InvalidPlan {
            reason: "CUDA launch dimensions must be nonzero",
        });
    }
    Ok(LaunchConfig {
        grid_dim: (elements.div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    })
}
