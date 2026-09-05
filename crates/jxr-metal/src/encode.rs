use jxr_core::device_plan::{OUTPUT_PLANE, SURFACE_HEIGHT, SURFACE_WIDTH};
use jxr_core::device_plan::{SAMPLE_OFFSET, SURFACE_OFFSET};
// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_metal_support::{
    checked_buffer_fill_bytes, checked_command_buffer, checked_compute_command_encoder,
    checked_event, checked_private_buffer, checked_shared_buffer_with_slice, dispatch_1d_pipeline,
    dispatch_2d_pipeline, mtl_size, one_d_threads_per_group,
};
use jxr_core::{OverlapMode, SurfaceLayout};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLEvent,
};

use crate::{
    MetalDecodePlan, MetalError,
    abi::{JxrPlaneAbi, macroblock_abi_metadata},
    buffer_pool::{MetalBufferPools, PooledBuffer},
    metal_types::JxrComputeEncoderExt,
    output_plan::{StorePipeline, build_output_dispatch},
    overlap_plan::{OverlapSchedule, first_overlap_schedule, second_overlap_schedule},
    plan::{MetalCoefficientSource, MetalReconstructionInput},
    runtime::MetalRuntime,
};

mod batch;

type BufferHandle = Retained<ProtocolObject<dyn MTLBuffer>>;
type BufferRef<'a> = &'a ProtocolObject<dyn MTLBuffer>;
type CommandHandle = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type ComputeEncoderHandle = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
type EventHandle = Retained<ProtocolObject<dyn MTLEvent>>;
type EncodingPair = (CommandHandle, ComputeEncoderHandle);

// Compatible images share descriptor and scratch arenas within this bounded
// command group. Keep the measured width explicit until batched final stores
// remove the per-image output-dispatch serialization.
const BATCH_IMAGES_PER_COMMAND: usize = 2;

struct ArenaBuffers {
    packed: BufferHandle,
    packed_offset_bytes: usize,
    macroblocks: BufferHandle,
}

struct ReconstructionBuffers {
    arenas: Vec<ArenaBuffers>,
    low: PooledBuffer,
    samples: PooledBuffer,
    status: PooledBuffer,
    output: BufferHandle,
    output_offset: usize,
}

pub(crate) struct EncodedMetalSubmission {
    pub(crate) command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub(crate) output: BufferHandle,
    pub(crate) status: BufferHandle,
    pub(crate) status_offset: usize,
    pub(crate) layout: SurfaceLayout,
    pub(crate) uploads: Vec<BufferHandle>,
    pub(crate) private_scratch: Vec<PooledBuffer>,
    pub(crate) shared_scratch: Vec<PooledBuffer>,
    pub(crate) buffer_pools: std::rc::Rc<MetalBufferPools>,
    pub(crate) completion_event: EventHandle,
}

struct PlanEncoding {
    buffers: ReconstructionBuffers,
    layout: SurfaceLayout,
}

pub(crate) fn encode(
    runtime: &MetalRuntime,
    plan: &MetalDecodePlan,
) -> Result<EncodedMetalSubmission, MetalError> {
    encode_with_output(runtime, plan, None)
}

pub(crate) fn encode_into(
    runtime: &MetalRuntime,
    plan: &MetalDecodePlan,
    output: BufferHandle,
) -> Result<EncodedMetalSubmission, MetalError> {
    encode_with_output(runtime, plan, Some((output, 0)))
}

pub(crate) fn encode_into_at(
    runtime: &MetalRuntime,
    plan: &MetalDecodePlan,
    output: BufferHandle,
    output_offset: usize,
) -> Result<EncodedMetalSubmission, MetalError> {
    encode_with_output(runtime, plan, Some((output, output_offset)))
}

pub(crate) fn encode_batch(
    runtime: &MetalRuntime,
    plans: &[MetalDecodePlan],
) -> Result<Vec<EncodedMetalSubmission>, MetalError> {
    encode_batch_with_outputs(runtime, plans, None)
}

pub(crate) fn encode_batch_into(
    runtime: &MetalRuntime,
    plans: &[MetalDecodePlan],
    outputs: &[BufferHandle],
) -> Result<Vec<EncodedMetalSubmission>, MetalError> {
    if plans.len() != outputs.len() {
        return Err(MetalError::InvalidPlan {
            reason: "Metal batch output count differs from plan count",
        });
    }
    encode_batch_with_outputs(runtime, plans, Some(outputs))
}

fn encode_batch_with_outputs(
    runtime: &MetalRuntime,
    plans: &[MetalDecodePlan],
    outputs: Option<&[BufferHandle]>,
) -> Result<Vec<EncodedMetalSubmission>, MetalError> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }
    let mut submissions = Vec::with_capacity(plans.len());
    for (group_index, group) in plans.chunks(BATCH_IMAGES_PER_COMMAND).enumerate() {
        let first = group_index * BATCH_IMAGES_PER_COMMAND;
        let queue = &runtime.batch_queues[group_index % runtime.batch_queues.len()];
        let group_outputs = outputs.map(|outputs| &outputs[first..first + group.len()]);
        if let Some(group_submissions) = batch::try_encode(runtime, queue, group, group_outputs)? {
            submissions.extend(group_submissions);
            continue;
        }
        let (command, encoder) =
            begin_encoding_on_queue(queue, "JXR concurrent main-profile batch reconstruction")?;
        let encodings = group
            .iter()
            .enumerate()
            .map(|(offset, plan)| {
                encode_plan(
                    runtime,
                    &encoder,
                    plan,
                    group_outputs.map(|outputs| (outputs[offset].clone(), 0)),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        encoder.endEncoding();
        let completion_event = checked_event(&queue.device())?;
        command.encodeSignalEvent_value(&completion_event, 1);
        command.commit();
        submissions.extend(encodings.into_iter().map(|encoding| {
            finish_encoding(runtime, command.clone(), completion_event.clone(), encoding)
        }));
    }
    Ok(submissions)
}

fn encode_with_output(
    runtime: &MetalRuntime,
    plan: &MetalDecodePlan,
    output: Option<(BufferHandle, usize)>,
) -> Result<EncodedMetalSubmission, MetalError> {
    let (command, encoder) = begin_encoding(runtime, "JXR main-profile reconstruction")?;
    let encoding = encode_plan(runtime, &encoder, plan, output)?;
    encoder.endEncoding();
    let completion_event = checked_event(&runtime.queue.device())?;
    command.encodeSignalEvent_value(&completion_event, 1);
    command.commit();
    Ok(finish_encoding(
        runtime,
        command,
        completion_event,
        encoding,
    ))
}

fn begin_encoding(runtime: &MetalRuntime, command_label: &str) -> Result<EncodingPair, MetalError> {
    begin_encoding_on_queue(&runtime.queue, command_label)
}

fn begin_encoding_on_queue(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    command_label: &str,
) -> Result<EncodingPair, MetalError> {
    let command = checked_command_buffer(queue)?;
    command.setLabel(Some(&NSString::from_str(command_label)));
    let encoder = checked_compute_command_encoder(&command)?;
    encoder.setLabel(Some(&NSString::from_str("JXR reconstruction phases")));
    Ok((command, encoder))
}

fn encode_plan(
    runtime: &MetalRuntime,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    plan: &MetalDecodePlan,
    output: Option<(BufferHandle, usize)>,
) -> Result<PlanEncoding, MetalError> {
    let input = plan.reconstruction()?;
    let buffers = allocate_buffers(runtime, plan, output)?;
    encode_low_pass(runtime, encoder, input, &buffers)?;
    encode_high_pass(runtime, encoder, input, &buffers)?;
    encode_output(runtime, encoder, plan, &buffers)?;
    Ok(PlanEncoding {
        buffers,
        layout: plan.output().clone(),
    })
}

fn encode_low_pass(
    runtime: &MetalRuntime,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &MetalReconstructionInput,
    buffers: &ReconstructionBuffers,
) -> Result<(), MetalError> {
    for &plane in input.planes.iter() {
        let plane_abi = JxrPlaneAbi::from_plan(plane)?;
        let arena = buffers
            .arenas
            .get(
                usize::try_from(plane.arena_index).map_err(|_| MetalError::InvalidPlan {
                    reason: "plane arena index exceeds usize",
                })?,
            )
            .ok_or(MetalError::InvalidPlan {
                reason: "plane references an absent coefficient arena",
            })?;
        encode_first_transform(
            encoder,
            runtime,
            arena,
            buffers,
            plane_abi,
            plane.macroblock_count,
        )?;
        barrier(encoder, buffers.low.buffer());
    }

    if input.overlap == OverlapMode::Two {
        for &plane in input.planes.iter() {
            let schedule = first_overlap_schedule(
                plane,
                input.hard_tiles,
                &input.tile_column_widths,
                &input.tile_row_heights,
            )?;
            encode_overlap_schedule(
                &runtime.queue.device(),
                encoder,
                &runtime.overlap_first,
                buffers.low.buffer(),
                buffers.status.buffer(),
                &schedule,
            )?;
        }
    }
    Ok(())
}

fn encode_high_pass(
    runtime: &MetalRuntime,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &MetalReconstructionInput,
    buffers: &ReconstructionBuffers,
) -> Result<(), MetalError> {
    for &plane in input.planes.iter() {
        let plane_abi = JxrPlaneAbi::from_plan(plane)?;
        let arena = &buffers.arenas[plane.arena_index as usize];
        encode_highpass(
            encoder,
            runtime,
            arena,
            buffers,
            plane_abi,
            plane.macroblock_count,
        )?;
        barrier(encoder, buffers.samples.buffer());
    }

    if input.overlap != OverlapMode::None {
        for &plane in input.planes.iter() {
            let schedule = second_overlap_schedule(
                plane,
                input.hard_tiles,
                &input.tile_column_widths,
                &input.tile_row_heights,
            )?;
            encode_overlap_schedule(
                &runtime.queue.device(),
                encoder,
                &runtime.overlap_second,
                buffers.samples.buffer(),
                buffers.status.buffer(),
                &schedule,
            )?;
        }
    }
    Ok(())
}

fn encode_output(
    runtime: &MetalRuntime,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    plan: &MetalDecodePlan,
    buffers: &ReconstructionBuffers,
) -> Result<(), MetalError> {
    encode_output_at(
        runtime,
        encoder,
        plan,
        buffers.samples.buffer(),
        buffers.status.buffer(),
        0,
        &buffers.output,
        buffers.output_offset,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_output_at(
    runtime: &MetalRuntime,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    plan: &MetalDecodePlan,
    samples: BufferRef<'_>,
    status: BufferRef<'_>,
    status_offset: usize,
    output: BufferRef<'_>,
    output_base: usize,
    sample_base: u32,
) -> Result<(), MetalError> {
    let device = runtime.queue.device();
    let mut output_dispatch = build_output_dispatch(plan)?;
    for plane in &mut output_dispatch.samples {
        plane[SAMPLE_OFFSET] =
            plane[SAMPLE_OFFSET]
                .checked_add(sample_base)
                .ok_or(MetalError::InvalidPlan {
                    reason: "batch sample descriptor offset exceeds u32",
                })?;
    }
    let output_base = u32::try_from(output_base).map_err(|_| MetalError::InvalidPlan {
        reason: "batch output base exceeds the Metal ABI",
    })?;
    for surface in &mut output_dispatch.surfaces {
        surface[SURFACE_OFFSET] =
            surface[SURFACE_OFFSET]
                .checked_add(output_base)
                .ok_or(MetalError::InvalidPlan {
                    reason: "batch output surface offset exceeds u32",
                })?;
    }
    let sample_planes = checked_shared_buffer_with_slice(&device, &output_dispatch.samples)?;
    let surface_planes = checked_shared_buffer_with_slice(&device, &output_dispatch.surfaces)?;
    let pipeline = runtime.output_store.select(output_dispatch.pipeline);
    let dispatch_count = if output_dispatch.planar {
        output_dispatch.surfaces.len()
    } else {
        1
    };
    for output_plane in 0..dispatch_count {
        let mut params = output_dispatch.params;
        params[OUTPUT_PLANE] =
            u32::try_from(output_plane).map_err(|_| MetalError::InvalidPlan {
                reason: "output plane index exceeds the Metal ABI",
            })?;
        let surface = output_dispatch.surfaces[output_plane];
        let dims = if output_dispatch.pipeline == StorePipeline::Bits {
            (surface[SURFACE_WIDTH].div_ceil(8), surface[SURFACE_HEIGHT])
        } else {
            (surface[SURFACE_WIDTH], surface[SURFACE_HEIGHT])
        };
        encoder.setComputePipelineState(pipeline);
        encoder.bind_buffer(0, samples, 0)?;
        encoder.bind_buffer(1, &sample_planes, 0)?;
        encoder.bind_buffer(2, &surface_planes, 0)?;
        encoder.bind_buffer(3, output, 0)?;
        encoder.bind_buffer(4, status, status_offset)?;
        encoder.bind_bytes(5, &params)?;
        dispatch_2d_pipeline(encoder, pipeline, dims);
    }
    Ok(())
}

fn finish_encoding(
    runtime: &MetalRuntime,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    completion_event: EventHandle,
    encoding: PlanEncoding,
) -> EncodedMetalSubmission {
    let buffers = encoding.buffers;
    let status = buffers.status.handle();
    let uploads = buffers
        .arenas
        .iter()
        .flat_map(|arena| [arena.packed.clone(), arena.macroblocks.clone()])
        .collect();
    EncodedMetalSubmission {
        command,
        output: buffers.output,
        status,
        status_offset: 0,
        layout: encoding.layout,
        uploads,
        private_scratch: vec![buffers.low, buffers.samples],
        shared_scratch: vec![buffers.status],
        buffer_pools: runtime.buffer_pools.clone(),
        completion_event,
    }
}

fn allocate_buffers(
    runtime: &MetalRuntime,
    plan: &MetalDecodePlan,
    output: Option<(BufferHandle, usize)>,
) -> Result<ReconstructionBuffers, MetalError> {
    let device = runtime.queue.device();
    let input = plan.reconstruction()?;
    let mut arenas = Vec::with_capacity(input.arenas.len());
    for arena in input.arenas.iter() {
        arenas.push(match &arena.source {
            MetalCoefficientSource::Cpu(coefficients) => {
                let upload = runtime.upload_cache.get_or_upload(&device, coefficients)?;
                ArenaBuffers {
                    packed: upload.packed,
                    packed_offset_bytes: 0,
                    macroblocks: upload.macroblocks,
                }
            }
            MetalCoefficientSource::Shared(coefficients) => {
                let packed_offset_bytes = coefficients
                    .element_offset()
                    .checked_mul(core::mem::size_of::<i32>())
                    .ok_or(MetalError::InvalidPlan {
                        reason: "shared coefficient byte offset overflows usize",
                    })?;
                ArenaBuffers {
                    packed: coefficients.buffer_handle(),
                    packed_offset_bytes,
                    macroblocks: checked_shared_buffer_with_slice(
                        &device,
                        &macroblock_abi_metadata(arena.macroblocks())?,
                    )?,
                }
            }
        });
    }
    let low_bytes = input
        .low_len
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or(MetalError::InvalidPlan {
            reason: "low-pass scratch byte count overflows usize",
        })?;
    let sample_bytes = input
        .sample_len
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or(MetalError::InvalidPlan {
            reason: "sample scratch byte count overflows usize",
        })?;
    let low = runtime.buffer_pools.take_private(&device, low_bytes)?;
    let samples = runtime.buffer_pools.take_private(&device, sample_bytes)?;
    let status = runtime
        .buffer_pools
        .take_shared(&device, core::mem::size_of::<u32>())?;
    // SAFETY: This new shared buffer has not been submitted or shared and the
    // initialized range covers its complete allocation.
    unsafe { checked_buffer_fill_bytes(status.buffer(), 0, core::mem::size_of::<u32>(), 0) }?;
    let (output, output_offset) = match output {
        Some((output, output_offset))
            if output_offset
                .checked_add(plan.output().byte_len)
                .is_some_and(|required| required <= output.length()) =>
        {
            (output, output_offset)
        }
        Some(_) => {
            return Err(MetalError::InvalidDestination {
                reason: "external output allocation is too small",
            });
        }
        None => (checked_private_buffer(&device, plan.output().byte_len)?, 0),
    };
    Ok(ReconstructionBuffers {
        arenas,
        low,
        samples,
        status,
        output,
        output_offset,
    })
}

fn encode_first_transform(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    runtime: &MetalRuntime,
    arena: &ArenaBuffers,
    buffers: &ReconstructionBuffers,
    plane: JxrPlaneAbi,
    macroblock_count: usize,
) -> Result<(), MetalError> {
    encoder.setComputePipelineState(&runtime.dequant_transform);
    encoder.bind_buffer(0, &arena.packed, arena.packed_offset_bytes)?;
    encoder.bind_buffer(1, &arena.macroblocks, 0)?;
    encoder.bind_buffer(2, buffers.low.buffer(), 0)?;
    encoder.bind_buffer(3, buffers.status.buffer(), 0)?;
    encoder.bind_bytes(4, &plane)?;
    dispatch_1d_pipeline(
        encoder,
        &runtime.dequant_transform,
        u64::try_from(macroblock_count).map_err(|_| MetalError::InvalidPlan {
            reason: "macroblock dispatch count exceeds u64",
        })?,
    );
    Ok(())
}

fn encode_highpass(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    runtime: &MetalRuntime,
    arena: &ArenaBuffers,
    buffers: &ReconstructionBuffers,
    plane: JxrPlaneAbi,
    macroblock_count: usize,
) -> Result<(), MetalError> {
    encoder.setComputePipelineState(&runtime.hp_transform);
    encoder.bind_buffer(0, &arena.packed, arena.packed_offset_bytes)?;
    encoder.bind_buffer(1, &arena.macroblocks, 0)?;
    encoder.bind_buffer(2, buffers.low.buffer(), 0)?;
    encoder.bind_buffer(3, buffers.samples.buffer(), 0)?;
    encoder.bind_buffer(4, buffers.status.buffer(), 0)?;
    encoder.bind_bytes(5, &plane)?;
    let width = (runtime.hp_transform.threadExecutionWidth() as u64).max(16);
    if width > runtime.hp_transform.maxTotalThreadsPerThreadgroup() as u64 {
        return Err(MetalError::InvalidPlan {
            reason: "Metal pipeline cannot host one transform macroblock",
        });
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        mtl_size(
            u64::try_from(macroblock_count).map_err(|_| MetalError::InvalidPlan {
                reason: "macroblock threadgroup count exceeds u64",
            })?,
            1,
            1,
        ),
        one_d_threads_per_group(width),
    );
    Ok(())
}

fn encode_overlap_schedule(
    device: &ProtocolObject<dyn MTLDevice>,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    samples: BufferRef<'_>,
    status: BufferRef<'_>,
    schedule: &OverlapSchedule,
) -> Result<(), MetalError> {
    encode_overlap_schedule_at(device, encoder, pipeline, samples, status, 0, schedule)
}

#[allow(clippy::too_many_arguments)]
fn encode_overlap_schedule_at(
    device: &ProtocolObject<dyn MTLDevice>,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    samples: BufferRef<'_>,
    status: BufferRef<'_>,
    status_offset: usize,
    schedule: &OverlapSchedule,
) -> Result<(), MetalError> {
    let has_work =
        !schedule.prefix.is_empty() || !schedule.filters.is_empty() || !schedule.suffix.is_empty();
    encode_overlap_list(
        device,
        encoder,
        pipeline,
        samples,
        status,
        status_offset,
        &schedule.prefix,
    )?;
    if !schedule.prefix.is_empty() {
        barrier(encoder, samples);
    }
    encode_overlap_list(
        device,
        encoder,
        pipeline,
        samples,
        status,
        status_offset,
        &schedule.filters,
    )?;
    if !schedule.filters.is_empty() && !schedule.suffix.is_empty() {
        barrier(encoder, samples);
    }
    encode_overlap_list(
        device,
        encoder,
        pipeline,
        samples,
        status,
        status_offset,
        &schedule.suffix,
    )?;
    if has_work {
        barrier(encoder, samples);
    }
    Ok(())
}

fn encode_overlap_list(
    device: &ProtocolObject<dyn MTLDevice>,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    samples: BufferRef<'_>,
    status: BufferRef<'_>,
    status_offset: usize,
    work: &[crate::abi::JxrOverlapWorkAbi],
) -> Result<(), MetalError> {
    if work.is_empty() {
        return Ok(());
    }
    let work_buffer = checked_shared_buffer_with_slice(device, work)?;
    let work_count = u32::try_from(work.len()).map_err(|_| MetalError::InvalidPlan {
        reason: "overlap work count exceeds the Metal ABI",
    })?;
    encoder.setComputePipelineState(pipeline);
    encoder.bind_buffer(0, samples, 0)?;
    encoder.bind_buffer(1, &work_buffer, 0)?;
    encoder.bind_buffer(2, status, status_offset)?;
    encoder.bind_bytes(3, &work_count)?;
    dispatch_1d_pipeline(encoder, pipeline, u64::from(work_count));
    Ok(())
}

fn barrier(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, buffer: BufferRef<'_>) {
    encoder.memory_barrier(&[buffer]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_metal_support::checked_shared_buffer;
    use jxr_core::{
        AlphaFormatRequest, AlphaMode, BackendRequest, BandPresence, BitstreamMode, ByteRange,
        ChannelLayout, ChromaSampling, CoefficientArena, CoefficientArenaDescriptor,
        CoefficientPlane, ColorFormat, CropWindow, DecodeScale, DecodedSamples, ImageInfo,
        ImageMetadata, MacroblockMetadata, OutputBitDepth, OutputFormatRequest, PixelFormat,
        PlaneInfo, PlanePlan, PredictionMode, PreparedPlan, QuantizerSet, Rect, SampleFormat,
        TileEdgeFlags, TileGrid, TilePlan,
    };
    use objc2_metal::MTLBlitCommandEncoder;
    use std::sync::Arc;

    fn single_plane_arena(dc: i32) -> Arc<CoefficientArena> {
        Arc::new(CoefficientArena {
            coefficients: vec![dc],
            macroblocks: MacroblockMetadata {
                coefficient_offsets: vec![0],
                quantizers: vec![QuantizerSet {
                    dc: 1,
                    low_pass: 1,
                    high_pass: 1,
                }],
                bands: vec![BandPresence::DcOnly],
                predictions: vec![PredictionMode::None],
                hp_predictions: vec![PredictionMode::None],
                tile_edges: vec![TileEdgeFlags::default()],
                coded_x: vec![0],
                coded_y: vec![0],
                output_x: vec![0],
                output_y: vec![0],
            },
            planes: vec![CoefficientPlane {
                coefficient_offset: 0,
                coefficient_count: 1,
                macroblock_offset: 0,
                macroblock_count: 1,
                block_columns: 4,
                block_rows: 4,
            }],
        })
    }

    fn full_plane_arena(dc_values: &[i32]) -> Arc<CoefficientArena> {
        let count = dc_values.len();
        Arc::new(CoefficientArena {
            coefficients: dc_values.to_vec(),
            macroblocks: MacroblockMetadata {
                coefficient_offsets: (0..u32::try_from(count).unwrap()).collect(),
                quantizers: vec![
                    QuantizerSet {
                        dc: 1,
                        low_pass: 1,
                        high_pass: 1,
                    };
                    count
                ],
                bands: vec![BandPresence::DcOnly; count],
                predictions: vec![PredictionMode::None; count],
                hp_predictions: vec![PredictionMode::None; count],
                tile_edges: vec![TileEdgeFlags::default(); count],
                coded_x: vec![0; count],
                coded_y: vec![0; count],
                output_x: vec![0; count],
                output_y: vec![0; count],
            },
            planes: (0..count)
                .map(|index| CoefficientPlane {
                    coefficient_offset: index,
                    coefficient_count: 1,
                    macroblock_offset: index,
                    macroblock_count: 1,
                    block_columns: 4,
                    block_rows: 4,
                })
                .collect(),
        })
    }

    fn luma_grid_arena() -> Arc<CoefficientArena> {
        Arc::new(CoefficientArena {
            coefficients: vec![0; 4],
            macroblocks: MacroblockMetadata {
                coefficient_offsets: vec![0, 1, 2, 3],
                quantizers: vec![
                    QuantizerSet {
                        dc: 1,
                        low_pass: 1,
                        high_pass: 1,
                    };
                    4
                ],
                bands: vec![BandPresence::DcOnly; 4],
                predictions: vec![PredictionMode::None; 4],
                hp_predictions: vec![PredictionMode::None; 4],
                tile_edges: vec![TileEdgeFlags::default(); 4],
                coded_x: vec![0, 1, 0, 1],
                coded_y: vec![0, 0, 1, 1],
                output_x: vec![0, 16, 0, 16],
                output_y: vec![0, 0, 16, 16],
            },
            planes: vec![CoefficientPlane {
                coefficient_offset: 0,
                coefficient_count: 4,
                macroblock_offset: 0,
                macroblock_count: 4,
                block_columns: 4,
                block_rows: 4,
            }],
        })
    }

    fn luma_plane_info() -> PlaneInfo {
        PlaneInfo {
            color_format: ColorFormat::Luma,
            sample_format: SampleFormat::Unsigned { bits: 8 },
            bands: BandPresence::DcOnly,
            bitstream_mode: BitstreamMode::Spatial,
            overlap: OverlapMode::None,
            short_header: false,
            long_word: false,
            scaled: false,
            chroma_centering: [0, 0],
            shift_bits: 0,
            mantissa_length: 0,
            exponent_bias: 0,
            width: 16,
            height: 16,
        }
    }

    fn one_macroblock_plan(info: ImageInfo) -> PreparedPlan {
        PreparedPlan {
            info,
            codestream_range: ByteRange {
                offset: 0,
                length: 1,
            },
            primary: PlanePlan {
                width: 16,
                height: 16,
                macroblocks_x: 1,
                macroblocks_y: 1,
                overlap: OverlapMode::None,
                coefficient_plane: 0,
            },
            alpha: None,
            tiles: vec![TilePlan {
                packet_range: ByteRange {
                    offset: 0,
                    length: 1,
                },
                output_region: Rect::full((16, 16)),
                macroblock_start: 0,
                macroblock_count: 1,
                hard_boundaries: false,
                required_for_reconstruction: true,
            }],
            reconstruction_region: Rect::full((16, 16)),
            output_region: Rect::full((16, 16)),
            decoded_region: Rect::full((16, 16)),
            scale: DecodeScale::Full,
            coefficient_bytes: 4,
        }
    }

    fn integrated_yuv420_info(color: ColorFormat) -> ImageInfo {
        let mut primary = luma_plane_info();
        primary.color_format = color;
        ImageInfo {
            width: 16,
            height: 16,
            profile: None,
            level: None,
            primary,
            alpha_mode: AlphaMode::Integrated,
            premultiplied_alpha: false,
            alpha: Some(luma_plane_info()),
            tiles: TileGrid {
                column_widths: vec![1],
                row_heights: vec![1],
                hard_tiles: false,
            },
            metadata: ImageMetadata::default(),
        }
    }

    fn integrated_yuv420_arena() -> Arc<CoefficientArena> {
        let plane = |index, block_columns, block_rows| CoefficientPlane {
            coefficient_offset: index,
            coefficient_count: 1,
            macroblock_offset: index,
            macroblock_count: 1,
            block_columns,
            block_rows,
        };
        Arc::new(CoefficientArena {
            coefficients: vec![16, 16, 0, 16],
            macroblocks: MacroblockMetadata {
                coefficient_offsets: vec![0, 1, 2, 3],
                quantizers: vec![
                    QuantizerSet {
                        dc: 1,
                        low_pass: 1,
                        high_pass: 1,
                    };
                    4
                ],
                bands: vec![BandPresence::DcOnly; 4],
                predictions: vec![PredictionMode::None; 4],
                hp_predictions: vec![PredictionMode::None; 4],
                tile_edges: vec![TileEdgeFlags::default(); 4],
                coded_x: vec![0; 4],
                coded_y: vec![0; 4],
                output_x: vec![0; 4],
                output_y: vec![0; 4],
            },
            planes: vec![
                plane(0, 4, 4),
                plane(1, 2, 2),
                plane(2, 2, 2),
                plane(3, 4, 4),
            ],
        })
    }

    fn rgba_policy(color: ColorFormat) -> OutputFormatRequest {
        OutputFormatRequest {
            internal_color: color,
            output_color: ColorFormat::Rgb,
            bit_depth: OutputBitDepth::U8,
            pixel_format: jxr_core::PixelFormat::U8(ChannelLayout::Rgba),
            scaled: false,
            alpha_format: Some(AlphaFormatRequest {
                bit_depth: OutputBitDepth::U8,
                scaled: false,
            }),
            red_blue_not_swapped: true,
            premultiply_alpha: false,
            crop: CropWindow {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
        }
    }

    fn assert_external_rgba(
        session: &crate::MetalDecoderSession,
        device: &ProtocolObject<dyn MTLDevice>,
        plan: &MetalDecodePlan,
    ) {
        let destination_buffer = checked_shared_buffer(device, plan.output().byte_len).unwrap();
        // SAFETY: the test owns the only handle used to access this fresh
        // allocation until the returned destination submission completes.
        let destination = unsafe {
            crate::MetalDestination::from_exclusive_buffer(
                destination_buffer.clone(),
                plan.output().clone(),
            )
            .unwrap()
        };
        let completion = session
            .submit_into(plan, destination)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(completion.layout, *plan.output());
        // SAFETY: waiting for completion ended exclusive GPU access, and the
        // checked read is bounded by the validated output layout.
        let external = unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(
                &destination_buffer,
                0,
                plan.output().byte_len,
            )
            .unwrap()
        };
        assert!(
            external
                .chunks_exact(4)
                .all(|pixel| pixel == [128, 130, 128, 129])
        );
    }

    fn assert_shared_rgba_and_planar(
        session: &crate::MetalDecoderSession,
        rgba_plan: &MetalDecodePlan,
        planar_plan: &MetalDecodePlan,
        expected_planar: &[u8],
    ) {
        let shared_batch = session
            .decode_batch_to_shared(&[rgba_plan.clone(), planar_plan.clone()])
            .unwrap();
        assert_eq!(shared_batch.len(), 2);
        shared_batch[0]
            .with_bytes(|bytes| {
                assert!(
                    bytes
                        .chunks_exact(4)
                        .all(|pixel| pixel == [128, 130, 128, 129])
                );
            })
            .unwrap();
        shared_batch[1]
            .with_bytes(|bytes| assert_eq!(bytes, expected_planar))
            .unwrap();
    }

    fn decode_luma(
        session: &crate::MetalDecoderSession,
        bit_depth: OutputBitDepth,
        pixel_format: PixelFormat,
    ) -> DecodedSamples {
        let plan = luma_plan(16, bit_depth, pixel_format);
        session.decode_to_host(&plan).unwrap().samples
    }

    fn luma_plan(dc: i32, bit_depth: OutputBitDepth, pixel_format: PixelFormat) -> MetalDecodePlan {
        let (prepared, policy, layout) = luma_plan_contract(bit_depth, pixel_format);
        MetalDecodePlan::from_prepared(
            single_plane_arena(dc),
            None,
            &prepared,
            policy,
            layout,
            [0, 0],
            BackendRequest::Metal,
        )
        .unwrap()
    }

    fn luma_plan_contract(
        bit_depth: OutputBitDepth,
        pixel_format: PixelFormat,
    ) -> (PreparedPlan, OutputFormatRequest, SurfaceLayout) {
        let info = ImageInfo {
            width: 16,
            height: 16,
            profile: None,
            level: None,
            primary: luma_plane_info(),
            alpha_mode: AlphaMode::None,
            premultiplied_alpha: false,
            alpha: None,
            tiles: TileGrid {
                column_widths: vec![1],
                row_heights: vec![1],
                hard_tiles: false,
            },
            metadata: ImageMetadata::default(),
        };
        let prepared = one_macroblock_plan(info);
        let policy = OutputFormatRequest {
            internal_color: ColorFormat::Luma,
            output_color: ColorFormat::Luma,
            bit_depth,
            pixel_format,
            scaled: false,
            alpha_format: None,
            red_blue_not_swapped: true,
            premultiply_alpha: false,
            crop: CropWindow {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
        };
        let layout = SurfaceLayout::for_output(policy, 1).unwrap();
        (prepared, policy, layout)
    }

    fn decode_color(
        session: &crate::MetalDecoderSession,
        color: ColorFormat,
        bit_depth: OutputBitDepth,
        pixel_format: PixelFormat,
        dc_values: &[i32],
    ) -> DecodedSamples {
        let mut primary = luma_plane_info();
        primary.color_format = color;
        let info = ImageInfo {
            width: 16,
            height: 16,
            profile: None,
            level: None,
            primary,
            alpha_mode: AlphaMode::None,
            premultiplied_alpha: false,
            alpha: None,
            tiles: TileGrid {
                column_widths: vec![1],
                row_heights: vec![1],
                hard_tiles: false,
            },
            metadata: ImageMetadata::default(),
        };
        let mut prepared = one_macroblock_plan(info);
        prepared.coefficient_bytes = core::mem::size_of_val(dc_values);
        let policy = OutputFormatRequest {
            internal_color: color,
            output_color: color,
            bit_depth,
            pixel_format,
            scaled: false,
            alpha_format: None,
            red_blue_not_swapped: true,
            premultiply_alpha: false,
            crop: CropWindow {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
        };
        let plan = MetalDecodePlan::from_prepared(
            full_plane_arena(dc_values),
            None,
            &prepared,
            policy,
            SurfaceLayout::for_output(policy, 1).unwrap(),
            [0, 0],
            BackendRequest::Metal,
        )
        .unwrap();
        session.decode_to_host(&plan).unwrap().samples
    }

    fn decode_overlap_grid(mode: OverlapMode, hard_tiles: bool) -> DecodedSamples {
        let mut primary = luma_plane_info();
        primary.overlap = mode;
        primary.width = 32;
        primary.height = 32;
        let info = ImageInfo {
            width: 32,
            height: 32,
            profile: None,
            level: None,
            primary,
            alpha_mode: AlphaMode::None,
            premultiplied_alpha: false,
            alpha: None,
            tiles: TileGrid {
                column_widths: vec![1, 1],
                row_heights: vec![1, 1],
                hard_tiles,
            },
            metadata: ImageMetadata::default(),
        };
        let tiles = (0..4)
            .map(|index| TilePlan {
                packet_range: ByteRange {
                    offset: 0,
                    length: 1,
                },
                output_region: Rect::full((32, 32)),
                macroblock_start: index,
                macroblock_count: 1,
                hard_boundaries: hard_tiles,
                required_for_reconstruction: true,
            })
            .collect();
        let prepared = PreparedPlan {
            info,
            codestream_range: ByteRange {
                offset: 0,
                length: 1,
            },
            primary: PlanePlan {
                width: 32,
                height: 32,
                macroblocks_x: 2,
                macroblocks_y: 2,
                overlap: mode,
                coefficient_plane: 0,
            },
            alpha: None,
            tiles,
            reconstruction_region: Rect::full((32, 32)),
            output_region: Rect::full((32, 32)),
            decoded_region: Rect::full((32, 32)),
            scale: DecodeScale::Full,
            coefficient_bytes: 16,
        };
        let policy = OutputFormatRequest {
            crop: CropWindow {
                x: 0,
                y: 0,
                width: 32,
                height: 32,
            },
            ..rgba_policy(ColorFormat::Luma)
        };
        let policy = OutputFormatRequest {
            output_color: ColorFormat::Luma,
            pixel_format: PixelFormat::U8(ChannelLayout::Luma),
            alpha_format: None,
            ..policy
        };
        let plan = MetalDecodePlan::from_prepared(
            luma_grid_arena(),
            None,
            &prepared,
            policy,
            SurfaceLayout::for_output(policy, 1).unwrap(),
            [0, 0],
            BackendRequest::Metal,
        )
        .unwrap();
        crate::MetalDecoderSession::system_default()
            .unwrap()
            .decode_to_host(&plan)
            .unwrap()
            .samples
    }

    #[test]
    fn dc_luma_macroblock_matches_scalar_u8_output() {
        let device = j2k_metal_support::system_default_device().unwrap();
        let runtime = MetalRuntime::build(&device, None).unwrap();
        let info = ImageInfo {
            width: 16,
            height: 16,
            profile: None,
            level: None,
            primary: luma_plane_info(),
            alpha_mode: AlphaMode::None,
            premultiplied_alpha: false,
            alpha: None,
            tiles: TileGrid {
                column_widths: vec![1],
                row_heights: vec![1],
                hard_tiles: false,
            },
            metadata: ImageMetadata::default(),
        };
        let prepared = one_macroblock_plan(info);
        let arena = single_plane_arena(16);
        let policy = OutputFormatRequest {
            internal_color: ColorFormat::Luma,
            output_color: ColorFormat::Luma,
            bit_depth: OutputBitDepth::U8,
            pixel_format: jxr_core::PixelFormat::U8(ChannelLayout::Luma),
            scaled: false,
            alpha_format: None,
            red_blue_not_swapped: true,
            premultiply_alpha: false,
            crop: CropWindow {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
        };
        let layout = SurfaceLayout::for_output(policy, 1).unwrap();
        let plan = MetalDecodePlan::from_prepared(
            arena.clone(),
            None,
            &prepared,
            policy,
            layout,
            [0, 0],
            BackendRequest::Metal,
        )
        .unwrap();
        let encoded = encode(&runtime, &plan).unwrap();
        j2k_metal_support::wait_for_completion(&encoded.command).unwrap();
        // SAFETY: the owning command completed and the status allocation is
        // shared, initialized, and large enough for one u32.
        let status =
            unsafe { j2k_metal_support::checked_buffer_read::<u32>(&encoded.status, 0).unwrap() };
        assert_eq!(status, 0);

        let shared = checked_shared_buffer(&device, 256).unwrap();
        let command = checked_command_buffer(&runtime.queue).unwrap();
        let blit = j2k_metal_support::checked_blit_command_encoder(&command).unwrap();
        // SAFETY: both buffers are retained through command completion and
        // the copied range fits their checked 256-byte allocations.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &encoded.output,
                0,
                &shared,
                0,
                256,
            );
        }
        blit.endEncoding();
        j2k_metal_support::commit_and_wait(&command).unwrap();
        // SAFETY: the blit completed and initialized the complete checked
        // shared allocation before this host read.
        let output =
            unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&shared, 0, 256).unwrap() };
        assert_eq!(output, vec![129; 256]);
    }

    #[test]
    fn concurrent_host_batch_preserves_caller_order_across_command_groups() {
        let plans = [0, 16, 32, 48, 64]
            .map(|dc| luma_plan(dc, OutputBitDepth::U8, PixelFormat::U8(ChannelLayout::Luma)));
        let images = crate::MetalDecoderSession::system_default()
            .unwrap()
            .decode_batch_to_host(&plans)
            .unwrap();
        assert_eq!(images.len(), plans.len());
        for (index, image) in images.into_iter().enumerate() {
            let DecodedSamples::U8(samples) = image.samples else {
                panic!("expected U8 batch output");
            };
            let expected = 128 + u8::try_from(index).unwrap();
            assert!(samples.iter().all(|&sample| sample == expected));
        }
    }

    #[test]
    fn shared_coefficient_batch_reads_each_direct_staging_slice() {
        let session = crate::MetalDecoderSession::system_default().unwrap();
        let staging = session.coefficient_staging_batch(1, 2).unwrap();
        let plans = staging
            .into_iter()
            .zip([0_i32, 16])
            .map(|(mut staging, dc)| {
                staging
                    .with_coefficients_mut(|coefficients| coefficients[0] = dc)
                    .unwrap();
                let arena = single_plane_arena(dc);
                let descriptor = CoefficientArenaDescriptor {
                    coefficient_count: arena.coefficients.len(),
                    macroblocks: arena.macroblocks.clone(),
                    planes: arena.planes.clone(),
                };
                let arena = Arc::new(staging.seal(descriptor).unwrap());
                let (prepared, policy, layout) =
                    luma_plan_contract(OutputBitDepth::U8, PixelFormat::U8(ChannelLayout::Luma));
                MetalDecodePlan::from_staged_primary(
                    arena,
                    &prepared,
                    policy,
                    layout,
                    [0, 0],
                    BackendRequest::Metal,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let images = session.decode_batch_to_host(&plans).unwrap();
        for (image, expected) in images.into_iter().zip([128_u8, 129]) {
            let DecodedSamples::U8(samples) = image.samples else {
                panic!("expected U8 batch output");
            };
            assert!(samples.iter().all(|&sample| sample == expected));
        }
    }

    #[test]
    fn luma_store_entrypoints_match_exact_scalar_values() {
        let session = crate::MetalDecoderSession::system_default().unwrap();
        let bit_white = decode_luma(
            &session,
            OutputBitDepth::Bit1White,
            PixelFormat::BitPacked(ChannelLayout::Luma),
        );
        assert!(matches!(bit_white, DecodedSamples::BitPacked(values) if values == vec![0xff; 32]));

        let bit_black = decode_luma(
            &session,
            OutputBitDepth::Bit1Black,
            PixelFormat::BitPacked(ChannelLayout::Luma),
        );
        assert!(matches!(bit_black, DecodedSamples::BitPacked(values) if values == vec![0; 32]));

        let u10 = decode_luma(
            &session,
            OutputBitDepth::U10,
            PixelFormat::U16(ChannelLayout::Luma),
        );
        assert!(matches!(u10, DecodedSamples::U16(values) if values.iter().all(|&v| v == 513)));

        let u16_samples = decode_luma(
            &session,
            OutputBitDepth::U16 { shift_bits: 0 },
            PixelFormat::U16(ChannelLayout::Luma),
        );
        assert!(
            matches!(u16_samples, DecodedSamples::U16(values) if values.iter().all(|&v| v == 32_769))
        );

        let i16_samples = decode_luma(
            &session,
            OutputBitDepth::I16 { shift_bits: 0 },
            PixelFormat::I16(ChannelLayout::Luma),
        );
        assert!(
            matches!(i16_samples, DecodedSamples::I16(values) if values.iter().all(|&v| v == 1))
        );

        let i32_samples = decode_luma(
            &session,
            OutputBitDepth::I32 { shift_bits: 0 },
            PixelFormat::I32(ChannelLayout::Luma),
        );
        assert!(
            matches!(i32_samples, DecodedSamples::I32(values) if values.iter().all(|&v| v == 1))
        );

        let f16_samples = decode_luma(
            &session,
            OutputBitDepth::F16,
            PixelFormat::F16(ChannelLayout::Luma),
        );
        assert!(
            matches!(f16_samples, DecodedSamples::F16(values) if values.iter().all(|&v| v == 1))
        );

        let f32_samples = decode_luma(
            &session,
            OutputBitDepth::F32 {
                mantissa_length: 0,
                exponent_bias: 1,
            },
            PixelFormat::F32(ChannelLayout::Luma),
        );
        assert!(
            matches!(f32_samples, DecodedSamples::F32(values) if values.iter().all(|&v| v.to_bits() == 1.0_f32.to_bits()))
        );
    }

    #[test]
    fn packed_rgb_rgbe_and_ncomponent_stores_cover_specialized_paths() {
        let session = crate::MetalDecoderSession::system_default().unwrap();
        let rgb555 = decode_color(
            &session,
            ColorFormat::Rgb,
            OutputBitDepth::Rgb555,
            PixelFormat::Rgb555,
            &[16; 3],
        );
        let expected_555 = 0x11 | (0x11 << 5) | (0x11 << 10);
        assert!(
            matches!(rgb555, DecodedSamples::Rgb555(values) if values.iter().all(|&v| v == expected_555))
        );

        let rgb565 = decode_color(
            &session,
            ColorFormat::Rgb,
            OutputBitDepth::Rgb565,
            PixelFormat::Rgb565,
            &[16; 3],
        );
        let expected_565 = 0x10 | (0x21 << 5) | (0x10 << 11);
        assert!(
            matches!(rgb565, DecodedSamples::Rgb565(values) if values.iter().all(|&v| v == expected_565))
        );

        let rgb101010 = decode_color(
            &session,
            ColorFormat::Rgb,
            OutputBitDepth::Rgb101010,
            PixelFormat::Rgb101010,
            &[16; 3],
        );
        let expected_101010 = 0x201 | (0x201 << 10) | (0x201 << 20);
        assert!(
            matches!(rgb101010, DecodedSamples::Rgb101010(values) if values.iter().all(|&v| v == expected_101010))
        );

        let asymmetric_rgb101010 = decode_color(
            &session,
            ColorFormat::Rgb,
            OutputBitDepth::Rgb101010,
            PixelFormat::Rgb101010,
            &[0, 16, 32],
        );
        let expected_asymmetric = 0x202 | (0x201 << 10) | (0x200 << 20);
        assert!(
            matches!(asymmetric_rgb101010, DecodedSamples::Rgb101010(values) if values.iter().all(|&v| v == expected_asymmetric))
        );

        let rgbe = decode_color(
            &session,
            ColorFormat::Rgbe,
            OutputBitDepth::U8,
            PixelFormat::Rgbe,
            &[16; 3],
        );
        assert!(
            matches!(rgbe, DecodedSamples::Rgbe(values) if values.iter().all(|&v| v == 0x0101_0101))
        );

        let ncomponent = decode_color(
            &session,
            ColorFormat::NComponent(5),
            OutputBitDepth::U8,
            PixelFormat::U8(ChannelLayout::NComponent(5)),
            &[16, 32, 48, 64, 80],
        );
        assert!(
            matches!(ncomponent, DecodedSamples::U8(values) if values.chunks_exact(5).all(|pixel| pixel == [129, 130, 131, 132, 133]))
        );
    }

    #[test]
    fn overlap_modes_preserve_zero_for_soft_and_hard_tile_regions() {
        for mode in [OverlapMode::One, OverlapMode::Two] {
            for hard_tiles in [false, true] {
                let output = decode_overlap_grid(mode, hard_tiles);
                assert!(
                    matches!(output, DecodedSamples::U8(values) if values.len() == 1_024 && values.iter().all(|&value| value == 128))
                );
            }
        }
    }

    #[test]
    fn arithmetic_failure_is_reported_after_submission_without_cpu_retry() {
        let info = ImageInfo {
            width: 16,
            height: 16,
            profile: None,
            level: None,
            primary: luma_plane_info(),
            alpha_mode: AlphaMode::None,
            premultiplied_alpha: false,
            alpha: None,
            tiles: TileGrid {
                column_widths: vec![1],
                row_heights: vec![1],
                hard_tiles: false,
            },
            metadata: ImageMetadata::default(),
        };
        let prepared = one_macroblock_plan(info);
        let policy = OutputFormatRequest {
            internal_color: ColorFormat::Luma,
            output_color: ColorFormat::Luma,
            bit_depth: OutputBitDepth::U8,
            pixel_format: PixelFormat::U8(ChannelLayout::Luma),
            scaled: false,
            alpha_format: None,
            red_blue_not_swapped: true,
            premultiply_alpha: false,
            crop: CropWindow {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
        };
        let plan = MetalDecodePlan::from_prepared(
            single_plane_arena(i32::MAX),
            None,
            &prepared,
            policy,
            SurfaceLayout::for_output(policy, 1).unwrap(),
            [0, 0],
            BackendRequest::Metal,
        )
        .unwrap();
        let error = crate::MetalDecoderSession::system_default()
            .unwrap()
            .submit(&plan)
            .unwrap()
            .wait()
            .unwrap_err();
        assert!(matches!(error, MetalError::KernelArithmetic { status } if status != 0));
    }

    #[test]
    fn integrated_alpha_yuv420_reconstructs_all_planes_to_rgba() {
        let color = ColorFormat::Yuv(ChromaSampling::Cs420);
        let mut prepared = one_macroblock_plan(integrated_yuv420_info(color));
        prepared.coefficient_bytes = 16;
        let arena = integrated_yuv420_arena();
        let policy = rgba_policy(color);
        let layout = SurfaceLayout::for_output(policy, 1).unwrap();
        let plan = MetalDecodePlan::from_prepared(
            arena.clone(),
            None,
            &prepared,
            policy,
            layout,
            [0, 0],
            BackendRequest::Metal,
        )
        .unwrap();
        let device = j2k_metal_support::system_default_device().unwrap();
        let session = crate::MetalDecoderSession::new(device.clone());
        let image = session.decode_to_host(&plan).unwrap();
        let DecodedSamples::U8(samples) = image.samples else {
            panic!("expected U8 Metal output");
        };
        assert_eq!(samples.len(), 16 * 16 * 4);
        assert!(
            samples
                .chunks_exact(4)
                .all(|pixel| pixel == [128, 130, 128, 129]),
            "unexpected first pixels: {:?}",
            &samples[..32]
        );

        let planar_policy = OutputFormatRequest {
            output_color: color,
            pixel_format: jxr_core::PixelFormat::U8(ChannelLayout::Yuva(ChromaSampling::Cs420)),
            ..policy
        };
        let planar_layout = SurfaceLayout::for_output(planar_policy, 1).unwrap();
        let planar_plan = MetalDecodePlan::from_prepared(
            arena,
            None,
            &prepared,
            planar_policy,
            planar_layout,
            [0, 0],
            BackendRequest::Metal,
        )
        .unwrap();
        let planar_image = session.decode_to_host(&planar_plan).unwrap();
        let DecodedSamples::U8(planar) = planar_image.samples else {
            panic!("expected planar U8 Metal output");
        };
        assert_eq!(planar.len(), 640);
        assert!(planar[..256].iter().all(|&sample| sample == 129));
        assert!(planar[256..320].iter().all(|&sample| sample == 130));
        assert!(planar[320..384].iter().all(|&sample| sample == 128));
        assert!(planar[384..].iter().all(|&sample| sample == 129));

        assert_external_rgba(&session, &device, &plan);

        let pending = session.submit(&plan).unwrap();
        assert_eq!(pending.output_layout().unwrap(), plan.output());
        // SAFETY: this observation does not access buffer contents; the
        // pending submission remains alive and no consumer is submitted.
        let pending_len = unsafe { pending.raw_output_buffer().unwrap().length() };
        assert_eq!(pending_len, plan.output().byte_len);
        let consumer_queue = j2k_metal_support::checked_command_queue(&device).unwrap();
        pending
            .enqueue_consumer_wait(&consumer_queue)
            .unwrap()
            .wait()
            .unwrap();
        let resident = pending.wait().unwrap();
        assert_eq!(resident.layout(), plan.output());

        let batch = session
            .submit_batch(&[plan.clone(), planar_plan.clone()])
            .unwrap();
        assert_eq!(batch.len(), 2);
        let resident = batch.wait().unwrap();
        assert_eq!(resident.len(), 2);
        let rgba_bytes = session.readback(&resident[0]).unwrap();
        assert!(
            rgba_bytes
                .chunks_exact(4)
                .all(|pixel| pixel == [128, 130, 128, 129])
        );
        assert_eq!(session.readback(&resident[1]).unwrap(), planar);

        let host_batch = session
            .decode_batch_to_host(&[plan.clone(), planar_plan.clone()])
            .unwrap();
        assert_eq!(host_batch.len(), 2);
        assert!(matches!(
            &host_batch[0].samples,
            DecodedSamples::U8(values)
                if values.chunks_exact(4).all(|pixel| pixel == [128, 130, 128, 129])
        ));
        assert_eq!(host_batch[1].samples, DecodedSamples::U8(planar.clone()));

        assert_shared_rgba_and_planar(&session, &plan, &planar_plan, &planar);
        let diagnostics = session.buffer_pool_diagnostics().unwrap();
        assert_eq!(diagnostics.private.cached_buffers, 4);
        assert_eq!(diagnostics.shared.cached_buffers, 4);
        let uploads = session.upload_cache_diagnostics().unwrap();
        assert_eq!(uploads.hits, 3);
        assert_eq!(uploads.misses, 1);
    }

    #[test]
    fn separately_encoded_alpha_is_reconstructed_from_its_own_arena() {
        let primary_info = luma_plane_info();
        let info = ImageInfo {
            width: 16,
            height: 16,
            profile: None,
            level: None,
            primary: primary_info.clone(),
            alpha_mode: AlphaMode::Separate,
            premultiplied_alpha: false,
            alpha: Some(luma_plane_info()),
            tiles: TileGrid {
                column_widths: vec![1],
                row_heights: vec![1],
                hard_tiles: false,
            },
            metadata: ImageMetadata::default(),
        };
        let mut prepared = one_macroblock_plan(info);
        prepared.alpha = Some(prepared.primary.clone());
        prepared.coefficient_bytes = 8;
        let alpha_info = ImageInfo {
            width: 16,
            height: 16,
            profile: None,
            level: None,
            primary: primary_info,
            alpha_mode: AlphaMode::None,
            premultiplied_alpha: false,
            alpha: None,
            tiles: TileGrid {
                column_widths: vec![1],
                row_heights: vec![1],
                hard_tiles: false,
            },
            metadata: ImageMetadata::default(),
        };
        let alpha_plan = one_macroblock_plan(alpha_info);
        let policy = OutputFormatRequest {
            internal_color: ColorFormat::Luma,
            output_color: ColorFormat::Luma,
            bit_depth: OutputBitDepth::U8,
            pixel_format: jxr_core::PixelFormat::U8(ChannelLayout::LumaAlpha),
            scaled: false,
            alpha_format: Some(AlphaFormatRequest {
                bit_depth: OutputBitDepth::U8,
                scaled: false,
            }),
            red_blue_not_swapped: true,
            premultiply_alpha: false,
            crop: CropWindow {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
        };
        let plan = MetalDecodePlan::from_prepared(
            single_plane_arena(16),
            Some((single_plane_arena(32), alpha_plan)),
            &prepared,
            policy,
            SurfaceLayout::for_output(policy, 1).unwrap(),
            [0, 0],
            BackendRequest::Metal,
        )
        .unwrap();
        let image = crate::MetalDecoderSession::system_default()
            .unwrap()
            .decode_to_host(&plan)
            .unwrap();
        let DecodedSamples::U8(samples) = image.samples else {
            panic!("expected U8 Metal output");
        };
        assert!(samples.chunks_exact(2).all(|pixel| pixel == [129, 130]));
    }
}
