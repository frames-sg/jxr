// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr::{DecodeReport, Rect};
use jxr_metal::{DenseMetalBatchLayout, MetalBatchDestination, MetalBatchDestinationCompletion};
use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::MPSGraphTensorData;

use crate::{
    Error, IndexedPreparationError, MpsGraphDecodeInput, MpsGraphPreparedBatch,
    MpsGraphPreparedGroup, MpsGraphTensorSpec,
    prepared::{IndexedGroupError, PreparedImage},
};
use crate::{MpsGraphProgram, MpsGraphRunOutput, SubmittedMpsGraphRun};

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type PreparedDestination = (
    Vec<jxr_metal::MetalDecodePlan>,
    DenseMetalBatchLayout,
    Retained<ProtocolObject<dyn MTLBuffer>>,
);

struct PreparedRequest {
    source_index: usize,
    image: jxr::PreparedJxr,
    request: jxr::DecodeRequest,
}

/// Persistent decoder whose JPEG XR and `MPSGraph` work share one Metal queue.
pub struct MpsGraphBatchDecoder {
    pub(super) session: jxr_metal::MetalDecoderSession,
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    batch_preparer: jxr::CpuBatchDecoder,
}

impl core::fmt::Debug for MpsGraphBatchDecoder {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MpsGraphBatchDecoder")
            .field("session", &self.session)
            .field("batch_preparer", &self.batch_preparer)
            .finish_non_exhaustive()
    }
}

impl MpsGraphBatchDecoder {
    /// Create a decoder on the system default Apple Silicon Metal device.
    pub fn system_default() -> Result<Self, Error> {
        Self::system_default_with_options(jxr::BatchDecodeOptions::default())
    }

    /// Create a decoder with retained shared native batch policy.
    pub fn system_default_with_options(options: jxr::BatchDecodeOptions) -> Result<Self, Error> {
        let device = j2k_metal_support::system_default_device()?;
        let queue = j2k_metal_support::checked_command_queue(&device)?;
        Self::with_device_and_queue_and_options(device, queue, options)
    }

    /// Create a decoder that shares an existing Metal device and command queue.
    pub fn with_device_and_queue(device: Device, queue: CommandQueue) -> Result<Self, Error> {
        Self::with_device_and_queue_and_options(device, queue, jxr::BatchDecodeOptions::default())
    }

    /// Create a decoder on an exact queue with retained native batch policy.
    pub fn with_device_and_queue_and_options(
        device: Device,
        queue: CommandQueue,
        batch_options: jxr::BatchDecodeOptions,
    ) -> Result<Self, Error> {
        if batch_options.layout != jxr::BatchLayout::Native {
            return Err(Error::Batch(
                jxr::BatchInfrastructureError::UnsupportedBatchLayout {
                    backend: "MPSGraph",
                    layout: batch_options.layout,
                },
            ));
        }
        let session =
            jxr_metal::MetalDecoderSession::with_command_queue(device.clone(), queue.clone())?;
        let batch_preparer = jxr::CpuBatchDecoder::new(batch_options)?;
        Ok(Self {
            session,
            device,
            queue,
            batch_preparer,
        })
    }

    /// Run CPU parsing/entropy preparation and partition valid inputs deterministically.
    pub fn prepare(
        &self,
        inputs: Vec<MpsGraphDecodeInput>,
    ) -> Result<MpsGraphPreparedBatch, Error> {
        let requests = inputs
            .into_iter()
            .enumerate()
            .map(|(source_index, input)| PreparedRequest {
                source_index,
                request: input.options.strict_request(),
                image: input.image,
            })
            .collect();
        Self::prepare_requests(requests, Vec::new())
    }

    /// Parse and group shared codec batch inputs without decoding coefficients.
    pub fn prepare_batch(
        &self,
        inputs: Vec<jxr::EncodedImage>,
    ) -> Result<jxr::PreparedBatch, Error> {
        Ok(self.batch_preparer.prepare(inputs)?)
    }

    /// Regroup shared prepared images without reparsing them.
    pub fn prepare_prepared_images(
        &self,
        images: Vec<jxr::PreparedImage>,
    ) -> Result<jxr::PreparedBatch, Error> {
        Ok(self.batch_preparer.prepare_prepared_images(images)?)
    }

    /// Decode a shared codec batch through the dense `MPSGraph` handoff.
    pub fn decode_batch(
        &mut self,
        prepared: &jxr::PreparedBatch,
    ) -> Result<MpsGraphBatchDecode, Error> {
        if prepared.input_count() > self.batch_preparer.options().max_inputs {
            return Err(Error::Batch(jxr::BatchInfrastructureError::TooManyInputs {
                requested: prepared.input_count(),
                maximum: self.batch_preparer.options().max_inputs,
            }));
        }
        let graph_prepared = Self::prepare_shared_batch(prepared)?;
        self.decode_prepared(&graph_prepared)
    }

    /// Regroup and decode shared prepared images without reparsing them.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<jxr::PreparedImage>,
    ) -> Result<MpsGraphBatchDecode, Error> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_batch(&prepared)
    }

    fn prepare_shared_batch(prepared: &jxr::PreparedBatch) -> Result<MpsGraphPreparedBatch, Error> {
        let mut errors = Vec::with_capacity(prepared.errors().len());
        for error in prepared.errors() {
            errors.push(IndexedPreparationError::new(
                error.index(),
                Error::Jxr(error.source().clone()),
            ));
        }
        let mut requests = Vec::new();
        for group in prepared.groups() {
            for (&source_index, image) in group.source_indices().iter().zip(group.images()) {
                if !matches!(
                    image.request().backend,
                    jxr::BackendRequest::Auto | jxr::BackendRequest::Metal
                ) {
                    errors.push(IndexedPreparationError::new(
                        source_index,
                        Error::Jxr(jxr::JxrError::new(
                            jxr::JxrErrorKind::BackendUnavailable,
                            "select MPSGraph batch decoder",
                        )),
                    ));
                    continue;
                }
                requests.push(PreparedRequest {
                    source_index,
                    image: image.image().clone(),
                    request: image
                        .request()
                        .clone()
                        .with_backend(jxr::BackendRequest::Metal),
                });
            }
        }
        requests.sort_by_key(|request| request.source_index);
        Self::prepare_requests(requests, errors)
    }

    fn prepare_requests(
        requests: Vec<PreparedRequest>,
        mut errors: Vec<IndexedPreparationError>,
    ) -> Result<MpsGraphPreparedBatch, Error> {
        let mut groups: Vec<MpsGraphPreparedGroup> = Vec::with_capacity(requests.len());
        for input in requests {
            let source_index = input.source_index;
            let prepared = (|| {
                let reconstruction = input
                    .image
                    .decoder()
                    .prepare_reconstruction(&input.request)?;
                let layout = reconstruction.output_layout();
                let individual_spec = MpsGraphTensorSpec::from_image_layout(layout, 1)?;
                DenseMetalBatchLayout::new(layout.clone(), 1)?;
                Ok::<_, Error>(PreparedImage {
                    source_index,
                    decoded_region: reconstruction.plan().output_region,
                    reconstruction,
                })
                .map(|image| (image, individual_spec))
            })();
            match prepared {
                Ok((image, individual_spec)) => {
                    if let Some(group) = groups.iter_mut().find(|group| {
                        group.spec.element_type() == individual_spec.element_type()
                            && group.spec.shape()[1..] == individual_spec.shape()[1..]
                    }) {
                        group.images.push(image);
                        group.spec = MpsGraphTensorSpec::from_image_layout(
                            group.images[0].reconstruction.output_layout(),
                            group.images.len(),
                        )?;
                    } else {
                        groups.push(MpsGraphPreparedGroup {
                            images: vec![image],
                            spec: individual_spec,
                        });
                    }
                }
                Err(error) => errors.push(IndexedPreparationError::new(source_index, error)),
            }
        }
        errors.sort_by_key(IndexedPreparationError::source_index);
        Ok(MpsGraphPreparedBatch { groups, errors })
    }

    /// Decode every valid prepared group into completed dense `MPSGraph` inputs.
    pub fn decode_prepared(
        &mut self,
        prepared: &MpsGraphPreparedBatch,
    ) -> Result<MpsGraphBatchDecode, Error> {
        let mut groups = Vec::with_capacity(prepared.groups.len());
        let mut group_errors = Vec::with_capacity(prepared.groups.len());
        for group in &prepared.groups {
            match self.decode_group(group) {
                Ok(decoded) => groups.push(decoded),
                Err(error) => {
                    group_errors.push(IndexedGroupError::new(group.source_indices(), error));
                }
            }
        }
        Ok(MpsGraphBatchDecode {
            groups,
            errors: prepared.errors.clone(),
            group_errors,
        })
    }

    fn decode_group(&mut self, group: &MpsGraphPreparedGroup) -> Result<MpsGraphInputGroup, Error> {
        let (plans, dense, buffer) = self.prepare_destination(group)?;
        // SAFETY: this adapter owns every handle to the fresh private buffer;
        // the returned completion retains exclusive access through completion.
        let destination =
            unsafe { MetalBatchDestination::from_exclusive_buffer(buffer.clone(), dense.clone())? };
        let completion = self
            .session
            .submit_batch_into(&plans, destination)?
            .wait()?;
        MpsGraphInputGroup::from_completed(&buffer, group, completion)
    }

    pub(super) fn prepare_destination(
        &self,
        group: &MpsGraphPreparedGroup,
    ) -> Result<PreparedDestination, Error> {
        let image_layout = group.images[0].reconstruction.output_layout().clone();
        let dense = DenseMetalBatchLayout::new(image_layout, group.images.len())?;
        if dense.byte_len() != group.spec.byte_len()? {
            return Err(Error::InvalidTensorContract {
                reason: "dense Metal batch length does not match its tensor contract",
            });
        }
        let mut plans = Vec::with_capacity(group.images.len());
        for image in &group.images {
            plans.push(image.reconstruction.metal_plan()?);
        }
        let buffer = j2k_metal_support::checked_private_buffer(&self.device, dense.byte_len())?;
        Ok((plans, dense, buffer))
    }

    /// Submit direct decode and graph execution on the shared queue without a CPU wait.
    pub fn submit_prepared_group(
        &mut self,
        program: &MpsGraphProgram,
        group: &MpsGraphPreparedGroup,
    ) -> Result<SubmittedMpsGraphRun, Error> {
        program.submit_prepared_group(self, group)
    }

    /// Submit direct decode and graph execution, then wait for completion.
    pub fn run_prepared_group(
        &mut self,
        program: &MpsGraphProgram,
        group: &MpsGraphPreparedGroup,
    ) -> Result<MpsGraphRunOutput, Error> {
        self.submit_prepared_group(program, group)?.wait()
    }

    #[must_use]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    #[must_use]
    pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }
}

/// One completed codec allocation wrapped as `MPSGraph` tensor data.
pub struct MpsGraphInputGroup {
    pub(super) tensor_data: Retained<MPSGraphTensorData>,
    pub(super) destination: MetalBatchDestination,
    pub(super) spec: MpsGraphTensorSpec,
    pub(super) source_indices: Vec<usize>,
    pub(super) decoded_regions: Vec<Rect>,
    pub(super) reports: Vec<DecodeReport>,
}

impl core::fmt::Debug for MpsGraphInputGroup {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MpsGraphInputGroup")
            .field("spec", &self.spec)
            .field("source_indices", &self.source_indices)
            .field("decoded_regions", &self.decoded_regions)
            .finish_non_exhaustive()
    }
}

impl MpsGraphInputGroup {
    fn from_completed(
        buffer: &ProtocolObject<dyn MTLBuffer>,
        group: &MpsGraphPreparedGroup,
        completion: MetalBatchDestinationCompletion,
    ) -> Result<Self, Error> {
        if completion.layout().byte_len() != group.spec.byte_len()? {
            return Err(Error::InvalidTensorContract {
                reason: "completed Metal batch length does not match its tensor contract",
            });
        }
        let tensor_data = tensor_data_from_buffer(buffer, group.spec);
        let mut source_indices = Vec::with_capacity(group.images.len());
        let mut decoded_regions = Vec::with_capacity(group.images.len());
        for image in &group.images {
            source_indices.push(image.source_index);
            decoded_regions.push(image.decoded_region);
        }
        let (destination, reports) = completion.into_parts();
        Ok(Self {
            tensor_data,
            destination,
            spec: group.spec,
            source_indices,
            decoded_regions,
            reports,
        })
    }

    #[must_use]
    pub fn tensor_data(&self) -> &MPSGraphTensorData {
        &self.tensor_data
    }

    #[must_use]
    pub const fn spec(&self) -> MpsGraphTensorSpec {
        self.spec
    }

    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    #[must_use]
    pub fn decoded_regions(&self) -> &[Rect] {
        &self.decoded_regions
    }

    #[must_use]
    pub fn reports(&self) -> &[DecodeReport] {
        &self.reports
    }
}

/// Successful completed groups plus isolated homogeneous execution failures.
#[derive(Debug)]
pub struct MpsGraphBatchDecode {
    groups: Vec<MpsGraphInputGroup>,
    errors: Vec<IndexedPreparationError>,
    group_errors: Vec<IndexedGroupError>,
}

impl MpsGraphBatchDecode {
    #[must_use]
    pub fn groups(&self) -> &[MpsGraphInputGroup] {
        &self.groups
    }

    #[must_use]
    pub fn errors(&self) -> &[IndexedPreparationError] {
        &self.errors
    }

    #[must_use]
    pub fn group_errors(&self) -> &[IndexedGroupError] {
        &self.group_errors
    }

    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Vec<MpsGraphInputGroup>,
        Vec<IndexedPreparationError>,
        Vec<IndexedGroupError>,
    ) {
        (self.groups, self.errors, self.group_errors)
    }
}

pub(super) fn tensor_data_from_buffer(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    spec: MpsGraphTensorSpec,
) -> Retained<MPSGraphTensorData> {
    let dimensions = spec.shape().map(NSNumber::new_usize);
    let shape = NSArray::from_retained_slice(&dimensions);
    // SAFETY: callers retain the buffer beyond tensor data, and `spec` proves
    // the dense allocation's static rank-four shape, dtype, and byte length.
    unsafe {
        MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
            MPSGraphTensorData::alloc(),
            buffer,
            &shape,
            spec.mps_data_type(),
        )
    }
}

impl MpsGraphTensorSpec {
    pub(super) const fn mps_data_type(self) -> MPSDataType {
        match self.element_type() {
            crate::MpsGraphElementType::U8 => MPSDataType::UInt8,
            crate::MpsGraphElementType::U16 => MPSDataType::UInt16,
            crate::MpsGraphElementType::I16 => MPSDataType::Int16,
        }
    }
}
