// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_metal_support::{
    checked_buffer_fill_bytes, checked_event, checked_private_buffer,
    checked_shared_buffer_with_slice, mtl_size, one_d_threads_per_group,
};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState,
};

use super::{
    BufferHandle, EncodedMetalSubmission, barrier, begin_encoding_on_queue, encode_output_at,
    encode_overlap_schedule_at,
};
use crate::{
    MetalDecodePlan, MetalError,
    abi::{JxrBatchDispatchAbi, JxrPlaneAbi, macroblock_abi_metadata},
    metal_types::JxrComputeEncoderExt,
    overlap_plan::{first_overlap_schedule, second_overlap_schedule},
    plan::{MetalCoefficientSource, MetalPlaneInput},
    runtime::MetalRuntime,
};

struct BatchBuffers {
    packed: BufferHandle,
    macroblocks: BufferHandle,
    planes: BufferHandle,
    low: crate::buffer_pool::PooledBuffer,
    samples: crate::buffer_pool::PooledBuffer,
    status: crate::buffer_pool::PooledBuffer,
    outputs: Vec<BufferHandle>,
    plane_inputs: Vec<Vec<MetalPlaneInput>>,
    sample_bases: Vec<u32>,
}

struct BatchDescriptors {
    packed: BufferHandle,
    macroblocks: BufferHandle,
    planes: BufferHandle,
    plane_inputs: Vec<Vec<MetalPlaneInput>>,
    sample_bases: Vec<u32>,
    low_len: usize,
    sample_len: usize,
}

pub(super) fn try_encode(
    runtime: &MetalRuntime,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    plans: &[MetalDecodePlan],
    outputs: Option<&[BufferHandle]>,
) -> Result<Option<Vec<EncodedMetalSubmission>>, MetalError> {
    if plans.len() < 2 || !compatible(plans)? {
        return Ok(None);
    }
    let (command, encoder) =
        begin_encoding_on_queue(queue, "JXR concatenated descriptor batch reconstruction")?;
    let buffers = build_buffers(runtime, plans, outputs)?;
    encode_first_transforms(runtime, &encoder, plans, &buffers)?;
    encode_first_overlaps(runtime, &encoder, plans, &buffers)?;
    encode_highpass_transforms(runtime, &encoder, plans, &buffers)?;
    encode_second_overlaps(runtime, &encoder, plans, &buffers)?;
    for (index, plan) in plans.iter().enumerate() {
        encode_output_at(
            runtime,
            &encoder,
            plan,
            buffers.samples.buffer(),
            buffers.status.buffer(),
            index * core::mem::size_of::<u32>(),
            &buffers.outputs[index],
            0,
            buffers.sample_bases[index],
        )?;
    }
    encoder.endEncoding();
    let completion_event = checked_event(&queue.device())?;
    command.encodeSignalEvent_value(&completion_event, 1);
    command.commit();
    Ok(Some(finish_batch(
        runtime,
        plans,
        &command,
        &completion_event,
        buffers,
    )))
}

fn compatible(plans: &[MetalDecodePlan]) -> Result<bool, MetalError> {
    let first = plans[0].reconstruction()?;
    if first.arenas.len() != 1 || first.planes.is_empty() {
        return Ok(false);
    }
    let first_allocation = match &first.arenas[0].source {
        MetalCoefficientSource::Cpu(_) => None,
        MetalCoefficientSource::Shared(arena) => Some(arena.allocation_address()),
    };
    for plan in &plans[1..] {
        let input = plan.reconstruction()?;
        if input.arenas.len() != 1 || input.planes.len() != first.planes.len() {
            return Ok(false);
        }
        let compatible_source = match (&first_allocation, &input.arenas[0].source) {
            (None, MetalCoefficientSource::Cpu(_)) => true,
            (Some(allocation), MetalCoefficientSource::Shared(arena)) => {
                *allocation == arena.allocation_address()
            }
            (None, MetalCoefficientSource::Shared(_))
            | (Some(_), MetalCoefficientSource::Cpu(_)) => false,
        };
        if !compatible_source {
            return Ok(false);
        }
    }
    Ok(true)
}

fn build_buffers(
    runtime: &MetalRuntime,
    plans: &[MetalDecodePlan],
    supplied_outputs: Option<&[BufferHandle]>,
) -> Result<BatchBuffers, MetalError> {
    let device = runtime.queue.device();
    let descriptors = build_descriptors(&device, plans)?;
    let low = runtime.buffer_pools.take_private(
        &device,
        byte_len_i32(descriptors.low_len, "batch low-pass")?,
    )?;
    let samples = runtime.buffer_pools.take_private(
        &device,
        byte_len_i32(descriptors.sample_len, "batch samples")?,
    )?;
    let status_bytes = plans
        .len()
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or_else(|| invalid("batch status length overflows usize"))?;
    let status = runtime.buffer_pools.take_shared(&device, status_bytes)?;
    // SAFETY: The pooled status allocation is exclusively held before submission.
    unsafe { checked_buffer_fill_bytes(status.buffer(), 0, status_bytes, 0) }?;
    let outputs = plans
        .iter()
        .enumerate()
        .map(|(index, plan)| {
            supplied_outputs.map_or_else(
                || {
                    checked_private_buffer(&device, plan.output().byte_len)
                        .map_err(MetalError::from)
                },
                |outputs| Ok(outputs[index].clone()),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(BatchBuffers {
        packed: descriptors.packed,
        macroblocks: descriptors.macroblocks,
        planes: descriptors.planes,
        low,
        samples,
        status,
        outputs,
        plane_inputs: descriptors.plane_inputs,
        sample_bases: descriptors.sample_bases,
    })
}

fn build_descriptors(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    plans: &[MetalDecodePlan],
) -> Result<BatchDescriptors, MetalError> {
    let mut packed = Vec::new();
    let shared_packed = match &plans[0].reconstruction()?.arenas[0].source {
        MetalCoefficientSource::Cpu(_) => None,
        MetalCoefficientSource::Shared(arena) => Some(arena.buffer_handle()),
    };
    let mut macroblocks = Vec::new();
    let mut plane_inputs = Vec::with_capacity(plans.len());
    let mut plane_abis = Vec::new();
    let mut sample_bases = Vec::with_capacity(plans.len());
    let mut low_len = 0_usize;
    let mut sample_len = 0_usize;
    for plan in plans {
        let input = plan.reconstruction()?;
        let arena = &input.arenas[0];
        let coefficient_base = match &arena.source {
            MetalCoefficientSource::Cpu(coefficients) => {
                let base = u32::try_from(packed.len())
                    .map_err(|_| invalid("concatenated coefficient offset exceeds u32"))?;
                packed.extend_from_slice(&coefficients.coefficients);
                base
            }
            MetalCoefficientSource::Shared(coefficients) => {
                u32::try_from(coefficients.element_offset())
                    .map_err(|_| invalid("shared coefficient offset exceeds u32"))?
            }
        };
        let macroblock_base = macroblocks.len();
        let mut metadata = macroblock_abi_metadata(arena.macroblocks())?;
        for macroblock in &mut metadata {
            macroblock.coefficient_offset = macroblock
                .coefficient_offset
                .checked_add(coefficient_base)
                .ok_or_else(|| invalid("concatenated coefficient reference exceeds u32"))?;
        }
        macroblocks.extend(metadata);
        let low_base = low_len;
        let sample_base = sample_len;
        sample_bases.push(
            u32::try_from(sample_base)
                .map_err(|_| invalid("concatenated sample base exceeds u32"))?,
        );
        let mut adjusted = Vec::with_capacity(input.planes.len());
        for &component_plane in input.planes.iter() {
            if component_plane.arena_index != 0 {
                return Err(invalid("compatible batch plane references another arena"));
            }
            let mut component_plane = component_plane;
            component_plane.macroblock_offset = component_plane
                .macroblock_offset
                .checked_add(macroblock_base)
                .ok_or_else(|| invalid("concatenated macroblock offset overflows usize"))?;
            component_plane.low_offset = component_plane
                .low_offset
                .checked_add(low_base)
                .ok_or_else(|| invalid("concatenated low-pass offset overflows usize"))?;
            component_plane.sample_offset = component_plane
                .sample_offset
                .checked_add(sample_base)
                .ok_or_else(|| invalid("concatenated sample offset overflows usize"))?;
            plane_abis.push(JxrPlaneAbi::from_plan(component_plane)?);
            adjusted.push(component_plane);
        }
        plane_inputs.push(adjusted);
        low_len = low_len
            .checked_add(input.low_len)
            .ok_or_else(|| invalid("concatenated low-pass length overflows usize"))?;
        sample_len = sample_len
            .checked_add(input.sample_len)
            .ok_or_else(|| invalid("concatenated sample length overflows usize"))?;
    }
    let packed = match shared_packed {
        Some(buffer) => buffer,
        None => checked_shared_buffer_with_slice(device, &packed)?,
    };
    Ok(BatchDescriptors {
        packed,
        macroblocks: checked_shared_buffer_with_slice(device, &macroblocks)?,
        planes: checked_shared_buffer_with_slice(device, &plane_abis)?,
        plane_inputs,
        sample_bases,
        low_len,
        sample_len,
    })
}

fn encode_first_transforms(
    runtime: &MetalRuntime,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    plans: &[MetalDecodePlan],
    buffers: &BatchBuffers,
) -> Result<(), MetalError> {
    let plane_count = buffers.plane_inputs[0].len();
    let work = buffers
        .plane_inputs
        .iter()
        .flat_map(|planes| planes.iter())
        .map(|plane| plane.macroblock_count)
        .max()
        .unwrap_or(0);
    let batch = batch_dispatch(plans.len(), plane_count)?;
    let width = runtime.batch_dequant_transform.threadExecutionWidth() as u64;
    encoder.setComputePipelineState(&runtime.batch_dequant_transform);
    encoder.bind_buffer(0, &buffers.packed, 0)?;
    encoder.bind_buffer(1, &buffers.macroblocks, 0)?;
    encoder.bind_buffer(2, buffers.low.buffer(), 0)?;
    encoder.bind_buffer(3, buffers.status.buffer(), 0)?;
    encoder.bind_buffer(4, &buffers.planes, 0)?;
    encoder.bind_bytes(5, &batch)?;
    encoder.dispatchThreads_threadsPerThreadgroup(
        mtl_size(
            u64::try_from(work).map_err(|_| invalid("batch first-transform work exceeds u64"))?,
            u64::from(batch.image_count),
            u64::from(batch.plane_count),
        ),
        one_d_threads_per_group(width),
    );
    barrier(encoder, buffers.low.buffer());
    Ok(())
}

fn encode_first_overlaps(
    runtime: &MetalRuntime,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    plans: &[MetalDecodePlan],
    buffers: &BatchBuffers,
) -> Result<(), MetalError> {
    for (image, plan) in plans.iter().enumerate() {
        let input = plan.reconstruction()?;
        if input.overlap != jxr_core::OverlapMode::Two {
            continue;
        }
        for &plane in &buffers.plane_inputs[image] {
            let schedule = first_overlap_schedule(
                plane,
                input.hard_tiles,
                &input.tile_column_widths,
                &input.tile_row_heights,
            )?;
            encode_overlap_schedule_at(
                &runtime.queue.device(),
                encoder,
                &runtime.overlap_first,
                buffers.low.buffer(),
                buffers.status.buffer(),
                image * core::mem::size_of::<u32>(),
                &schedule,
            )?;
        }
    }
    Ok(())
}

fn encode_highpass_transforms(
    runtime: &MetalRuntime,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    plans: &[MetalDecodePlan],
    buffers: &BatchBuffers,
) -> Result<(), MetalError> {
    let plane_count = buffers.plane_inputs[0].len();
    let width = (runtime.batch_hp_transform.threadExecutionWidth() as u64).max(16);
    if width > runtime.batch_hp_transform.maxTotalThreadsPerThreadgroup() as u64 {
        return Err(invalid(
            "batch HP pipeline cannot host one transform macroblock",
        ));
    }
    let work = buffers
        .plane_inputs
        .iter()
        .flat_map(|planes| planes.iter())
        .map(|plane| plane.macroblock_count)
        .max()
        .unwrap_or(0);
    let batch = batch_dispatch(plans.len(), plane_count)?;
    encoder.setComputePipelineState(&runtime.batch_hp_transform);
    encoder.bind_buffer(0, &buffers.packed, 0)?;
    encoder.bind_buffer(1, &buffers.macroblocks, 0)?;
    encoder.bind_buffer(2, buffers.low.buffer(), 0)?;
    encoder.bind_buffer(3, buffers.samples.buffer(), 0)?;
    encoder.bind_buffer(4, buffers.status.buffer(), 0)?;
    encoder.bind_buffer(5, &buffers.planes, 0)?;
    encoder.bind_bytes(6, &batch)?;
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        mtl_size(
            u64::try_from(work).map_err(|_| invalid("batch HP work exceeds u64"))?,
            u64::from(batch.image_count),
            u64::from(batch.plane_count),
        ),
        one_d_threads_per_group(width),
    );
    barrier(encoder, buffers.samples.buffer());
    Ok(())
}

fn encode_second_overlaps(
    runtime: &MetalRuntime,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    plans: &[MetalDecodePlan],
    buffers: &BatchBuffers,
) -> Result<(), MetalError> {
    for (image, plan) in plans.iter().enumerate() {
        let input = plan.reconstruction()?;
        if input.overlap == jxr_core::OverlapMode::None {
            continue;
        }
        for &plane in &buffers.plane_inputs[image] {
            let schedule = second_overlap_schedule(
                plane,
                input.hard_tiles,
                &input.tile_column_widths,
                &input.tile_row_heights,
            )?;
            encode_overlap_schedule_at(
                &runtime.queue.device(),
                encoder,
                &runtime.overlap_second,
                buffers.samples.buffer(),
                buffers.status.buffer(),
                image * core::mem::size_of::<u32>(),
                &schedule,
            )?;
        }
    }
    Ok(())
}

fn finish_batch(
    runtime: &MetalRuntime,
    plans: &[MetalDecodePlan],
    command: &super::CommandHandle,
    completion_event: &super::EventHandle,
    mut buffers: BatchBuffers,
) -> Vec<EncodedMetalSubmission> {
    let status = buffers.status.handle();
    let mut low = Some(buffers.low);
    let mut samples = Some(buffers.samples);
    let mut status_owner = Some(buffers.status);
    plans
        .iter()
        .zip(buffers.outputs.drain(..))
        .enumerate()
        .map(|(index, (plan, output))| EncodedMetalSubmission {
            command: command.clone(),
            output,
            status: status.clone(),
            status_offset: index * core::mem::size_of::<u32>(),
            layout: plan.output().clone(),
            uploads: if index == 0 {
                vec![
                    buffers.packed.clone(),
                    buffers.macroblocks.clone(),
                    buffers.planes.clone(),
                ]
            } else {
                Vec::new()
            },
            private_scratch: if index == 0 {
                vec![
                    low.take().expect("first batch output owns low scratch"),
                    samples
                        .take()
                        .expect("first batch output owns sample scratch"),
                ]
            } else {
                Vec::new()
            },
            shared_scratch: status_owner.take().into_iter().collect(),
            buffer_pools: runtime.buffer_pools.clone(),
            completion_event: completion_event.clone(),
        })
        .collect()
}

fn batch_dispatch(images: usize, planes: usize) -> Result<JxrBatchDispatchAbi, MetalError> {
    Ok(JxrBatchDispatchAbi {
        image_count: u32_count(images, "batch image count")?,
        plane_count: u32_count(planes, "batch plane count")?,
        plane_index: 0,
        reserved: 0,
    })
}

fn byte_len_i32(elements: usize, name: &'static str) -> Result<usize, MetalError> {
    elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| invalid(name))
}

fn u32_count(value: usize, reason: &'static str) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| invalid(reason))
}

const fn invalid(reason: &'static str) -> MetalError {
    MetalError::InvalidPlan { reason }
}
