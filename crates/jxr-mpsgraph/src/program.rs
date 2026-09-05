// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_mpsgraph_support::{GraphExecutionError, MpsGraphSubmission};
use jxr::{DecodeReport, Rect};
use jxr_metal::{MetalBatchDestination, MetalBatchDestinationSubmission};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{MPSGraph, MPSGraphTensor, MPSGraphTensorData};

use crate::{
    Error, MpsGraphInputGroup, MpsGraphPreparedGroup, MpsGraphTensorSpec,
    platform::{MpsGraphBatchDecoder, tensor_data_from_buffer},
};

/// Static rank-four `MPSGraph` program with one image placeholder.
pub struct MpsGraphProgram {
    graph: Retained<MPSGraph>,
    image_placeholder: Retained<MPSGraphTensor>,
    targets: Vec<Retained<MPSGraphTensor>>,
    input_spec: MpsGraphTensorSpec,
}

impl core::fmt::Debug for MpsGraphProgram {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MpsGraphProgram")
            .field("input_spec", &self.input_spec)
            .field("target_count", &self.targets.len())
            .finish_non_exhaustive()
    }
}

impl MpsGraphProgram {
    /// Adopt a graph, its sole runtime image placeholder, and output targets.
    ///
    /// Other graph inputs must be constants embedded in `graph`.
    pub fn new(
        graph: Retained<MPSGraph>,
        image_placeholder: Retained<MPSGraphTensor>,
        targets: Vec<Retained<MPSGraphTensor>>,
        input_spec: MpsGraphTensorSpec,
    ) -> Result<Self, Error> {
        if targets.is_empty() {
            return Err(Error::InvalidTensorContract {
                reason: "MPSGraph program requires at least one target tensor",
            });
        }
        validate_placeholder(&image_placeholder, input_spec)?;
        Ok(Self {
            graph,
            image_placeholder,
            targets,
            input_spec,
        })
    }

    /// Build an identity graph for direct handoff validation.
    pub fn identity(input_spec: MpsGraphTensorSpec) -> Result<Self, Error> {
        // SAFETY: `new` is a standard owning Objective-C constructor.
        let graph = unsafe { MPSGraph::new() };
        let shape = mps_shape(input_spec.shape());
        // SAFETY: shape and dtype are validated static values retained by the graph.
        let placeholder = unsafe {
            graph.placeholderWithShape_dataType_name(Some(&shape), input_spec.mps_data_type(), None)
        };
        let targets = vec![placeholder.clone()];
        Self::new(graph, placeholder, targets, input_spec)
    }

    /// Build the RGB8/NHWC normalization, weighting, and reduction reference graph.
    pub fn rgb8_nhwc_reference(batch: usize, height: usize, width: usize) -> Result<Self, Error> {
        let input_spec =
            MpsGraphTensorSpec::new([batch, height, width, 3], crate::MpsGraphElementType::U8)?;
        let spatial_pixels = height
            .checked_mul(width)
            .and_then(|pixels| u32::try_from(pixels).ok())
            .ok_or(Error::TensorShapeOverflow)?;
        // SAFETY: `new` is a standard owning Objective-C constructor.
        let graph = unsafe { MPSGraph::new() };
        let shape = mps_shape(input_spec.shape());
        // SAFETY: all operations have static shapes, valid axes, and graph-owned constants.
        let (placeholder, score) = unsafe {
            let placeholder =
                graph.placeholderWithShape_dataType_name(Some(&shape), MPSDataType::UInt8, None);
            let float = graph.castTensor_toType_name(&placeholder, MPSDataType::Float32, None);
            let scale = graph.constantWithScalar_dataType(255.0, MPSDataType::Float32);
            let normalized =
                graph.divisionWithPrimaryTensor_secondaryTensor_name(&float, &scale, None);
            let weighted_channel = |channel: isize, weight: f64| {
                let values =
                    graph.sliceTensor_dimension_start_length_name(&normalized, 3, channel, 1, None);
                let coefficient = graph.constantWithScalar_dataType(weight, MPSDataType::Float32);
                graph.multiplicationWithPrimaryTensor_secondaryTensor_name(
                    &values,
                    &coefficient,
                    None,
                )
            };
            let red = weighted_channel(0, f64::from(crate::RGB8_REFERENCE_CHANNEL_WEIGHTS[0]));
            let green = weighted_channel(1, f64::from(crate::RGB8_REFERENCE_CHANNEL_WEIGHTS[1]));
            let blue = weighted_channel(2, f64::from(crate::RGB8_REFERENCE_CHANNEL_WEIGHTS[2]));
            let red_green =
                graph.additionWithPrimaryTensor_secondaryTensor_name(&red, &green, None);
            let weighted =
                graph.additionWithPrimaryTensor_secondaryTensor_name(&red_green, &blue, None);
            let axes = NSArray::from_retained_slice(&[
                NSNumber::new_isize(1),
                NSNumber::new_isize(2),
                NSNumber::new_isize(3),
            ]);
            let summed = graph.reductionSumWithTensor_axes_name(&weighted, Some(&axes), None);
            let count =
                graph.constantWithScalar_dataType(f64::from(spatial_pixels), MPSDataType::Float32);
            let score = graph.divisionWithPrimaryTensor_secondaryTensor_name(&summed, &count, None);
            (placeholder, score)
        };
        Self::new(graph, placeholder, vec![score], input_spec)
    }

    #[must_use]
    pub const fn input_spec(&self) -> MpsGraphTensorSpec {
        self.input_spec
    }

    #[must_use]
    pub fn graph(&self) -> &MPSGraph {
        &self.graph
    }

    #[must_use]
    pub fn image_placeholder(&self) -> &MPSGraphTensor {
        &self.image_placeholder
    }

    #[must_use]
    pub fn targets(&self) -> &[Retained<MPSGraphTensor>] {
        &self.targets
    }

    /// Reject a group whose shape or native dtype differs from this program.
    pub fn validate_input_spec(&self, actual: MpsGraphTensorSpec) -> Result<(), Error> {
        if actual != self.input_spec {
            return Err(Error::InvalidTensorContract {
                reason: "MPSGraph image placeholder shape or dtype does not match the batch",
            });
        }
        Ok(())
    }

    /// Submit graph execution for a completed dense input group.
    pub fn submit_completed(
        &self,
        command_queue: &ProtocolObject<dyn MTLCommandQueue>,
        input: MpsGraphInputGroup,
    ) -> Result<SubmittedMpsGraphRun, Error> {
        self.validate_input_spec(input.spec)?;
        let MpsGraphInputGroup {
            tensor_data,
            destination,
            spec: _,
            source_indices,
            decoded_regions,
            reports,
        } = input;
        let feed_data = tensor_data.clone();
        let metadata = RunMetadata {
            source_indices,
            completed: Some((decoded_regions, reports)),
        };
        Ok(self.submit_graph(
            command_queue,
            &feed_data,
            RunInputOwner::Completed {
                tensor_data,
                destination,
            },
            None,
            metadata,
        ))
    }

    pub(crate) fn submit_prepared_group(
        &self,
        decoder: &mut MpsGraphBatchDecoder,
        group: &MpsGraphPreparedGroup,
    ) -> Result<SubmittedMpsGraphRun, Error> {
        self.validate_input_spec(group.spec())?;
        let (plans, dense, buffer) = decoder.prepare_destination(group)?;
        // SAFETY: the adapter owns every handle to the fresh private buffer;
        // codec writes and graph reads are ordered on the same retained queue.
        let destination =
            unsafe { MetalBatchDestination::from_exclusive_buffer(buffer.clone(), dense)? };
        let codec = decoder.session.submit_batch_into(&plans, destination)?;
        let tensor_data = tensor_data_from_buffer(&buffer, group.spec());
        let mut source_indices = Vec::with_capacity(group.images.len());
        let mut decoded_regions = Vec::with_capacity(group.images.len());
        for image in &group.images {
            source_indices.push(image.source_index);
            decoded_regions.push(image.decoded_region);
        }
        let metadata = RunMetadata {
            source_indices,
            completed: Some((decoded_regions, Vec::new())),
        };
        Ok(self.submit_graph(
            &decoder.queue,
            &tensor_data,
            RunInputOwner::Direct {
                tensor_data: tensor_data.clone(),
                buffer,
            },
            Some(codec),
            metadata,
        ))
    }

    fn submit_graph(
        &self,
        command_queue: &ProtocolObject<dyn MTLCommandQueue>,
        tensor_data: &MPSGraphTensorData,
        input_owner: RunInputOwner,
        codec: Option<MetalBatchDestinationSubmission>,
        metadata: RunMetadata,
    ) -> SubmittedMpsGraphRun {
        // SAFETY: the program image contract and queue device were checked
        // before submission. Codec writes precede graph reads on this same queue.
        // RunInputOwner retains tensor data and its standalone buffer or resident
        // destination, and the shared guard retains them through graph completion.
        let graph = unsafe {
            MpsGraphSubmission::submit(
                &self.graph,
                &self.image_placeholder,
                &self.targets,
                command_queue,
                tensor_data,
                input_owner,
            )
        };
        SubmittedMpsGraphRun {
            graph,
            codec,
            metadata: Some(metadata),
        }
    }
}

fn validate_placeholder(
    placeholder: &MPSGraphTensor,
    expected: MpsGraphTensorSpec,
) -> Result<(), Error> {
    // SAFETY: immutable metadata remains valid while the tensor is retained.
    let shape = unsafe { placeholder.shape() }.ok_or(Error::InvalidTensorContract {
        reason: "MPSGraph image placeholder must have a static rank-four shape",
    })?;
    if shape.len() != 4 {
        return Err(Error::InvalidTensorContract {
            reason: "MPSGraph image placeholder must have rank four",
        });
    }
    let actual = core::array::from_fn(|index| shape.objectAtIndex(index).as_usize());
    if actual != expected.shape() {
        return Err(Error::InvalidTensorContract {
            reason: "MPSGraph placeholder shape does not match its contract",
        });
    }
    // SAFETY: immutable dtype metadata is valid for the retained tensor.
    if unsafe { placeholder.dataType() } != expected.mps_data_type() {
        return Err(Error::InvalidTensorContract {
            reason: "MPSGraph placeholder dtype does not match its contract",
        });
    }
    Ok(())
}

fn mps_shape(shape: [usize; 4]) -> Retained<NSArray<NSNumber>> {
    let dimensions = shape.map(NSNumber::new_usize);
    NSArray::from_retained_slice(&dimensions)
}

struct RunMetadata {
    source_indices: Vec<usize>,
    completed: Option<(Vec<Rect>, Vec<DecodeReport>)>,
}

#[expect(
    dead_code,
    reason = "fields are lifetime guards released after completion"
)]
enum RunInputOwner {
    Completed {
        tensor_data: Retained<MPSGraphTensorData>,
        destination: MetalBatchDestination,
    },
    Direct {
        tensor_data: Retained<MPSGraphTensorData>,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    },
}

/// Completed graph outputs and JPEG XR metadata for their input batch.
pub struct MpsGraphRunOutput {
    results: Vec<Retained<MPSGraphTensorData>>,
    source_indices: Vec<usize>,
    decoded_regions: Vec<Rect>,
    reports: Vec<DecodeReport>,
}

impl core::fmt::Debug for MpsGraphRunOutput {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MpsGraphRunOutput")
            .field("result_count", &self.results.len())
            .field("source_indices", &self.source_indices)
            .field("decoded_regions", &self.decoded_regions)
            .finish_non_exhaustive()
    }
}

impl MpsGraphRunOutput {
    #[must_use]
    pub fn results(&self) -> &[Retained<MPSGraphTensorData>] {
        &self.results
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

/// In-flight direct decode and `MPSGraph` execution.
///
/// This guard is deliberately neither `Send` nor `Sync`. Dropping it waits
/// before releasing any unretained Metal storage.
pub struct SubmittedMpsGraphRun {
    graph: MpsGraphSubmission<RunInputOwner>,
    codec: Option<MetalBatchDestinationSubmission>,
    metadata: Option<RunMetadata>,
}

impl core::fmt::Debug for SubmittedMpsGraphRun {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SubmittedMpsGraphRun")
            .field("complete", &self.is_complete())
            .finish_non_exhaustive()
    }
}

impl SubmittedMpsGraphRun {
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.graph.is_complete()
    }

    /// Wait for both graph and codec completion and return target tensors.
    pub fn wait(mut self) -> Result<MpsGraphRunOutput, Error> {
        self.finish()
    }

    fn finish(&mut self) -> Result<MpsGraphRunOutput, Error> {
        let graph_error = self.graph.wait().err();
        let completion = self
            .codec
            .take()
            .map(MetalBatchDestinationSubmission::wait)
            .transpose()?;
        let mut metadata = self
            .metadata
            .take()
            .expect("MPSGraph run metadata is consumed exactly once");
        if let Some(completion) = completion {
            let (regions, _) = metadata
                .completed
                .take()
                .expect("direct run records decoded regions before submission");
            let (_destination, reports) = completion.into_parts();
            metadata.completed = Some((regions, reports));
        }
        if let Some(error) = graph_error {
            return Err(graph_execution_error(error));
        }
        let mut results = Vec::with_capacity(self.graph.target_count());
        for index in 0..self.graph.target_count() {
            let result = self
                .graph
                .output(index)
                .map_err(graph_execution_error)?
                .ok_or(Error::MissingGraphOutput { index })?;
            results.push(result);
        }
        let (decoded_regions, reports) = metadata
            .completed
            .take()
            .expect("codec metadata exists after completion");
        Ok(MpsGraphRunOutput {
            results,
            source_indices: metadata.source_indices,
            decoded_regions,
            reports,
        })
    }
}

impl Drop for SubmittedMpsGraphRun {
    fn drop(&mut self) {
        // Cleanup waits without extracting metadata or allocating output vectors.
        let _ = self.graph.wait();
        if let Some(codec) = self.codec.take() {
            let _ = codec.wait();
        }
    }
}

fn graph_execution_error(error: GraphExecutionError) -> Error {
    Error::GraphExecution {
        domain: error.domain,
        code: error.code,
        description: error.description,
    }
}
