// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use jxr_core::{
    AlphaMode, BackendKind, BackendRequest, ChromaSampling, CoefficientArena, CoefficientPlane,
    ColorFormat, DecodeReport, DecodeStage, ImageInfo, MacroblockMetadata, OutputFormatRequest,
    OverlapMode, PredictionMode, PreparedPlan, Rect, StageExecutor, StageReport, SurfaceLayout,
};

use crate::CudaError;

const INTERNAL_ALIGNMENT_I32: usize = 64;

/// Validated CUDA reconstruction plan retaining CPU-produced coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDecodePlan {
    reconstructed_coefficients: u64,
    coefficient_bytes: usize,
    output: SurfaceLayout,
    info: Option<ImageInfo>,
    output_region: Option<Rect>,
    output_policy: Option<OutputFormatRequest>,
    requested_backend: BackendRequest,
    reconstruction: Option<CudaReconstructionInput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CudaArenaInput {
    pub(crate) source: Arc<CoefficientArena>,
    pub(crate) overlap: OverlapMode,
    pub(crate) hard_tiles: bool,
    pub(crate) tile_column_widths: Arc<[u32]>,
    pub(crate) tile_row_heights: Arc<[u32]>,
}

pub(crate) use jxr_core::device_plan::ReconstructionPlane as CudaPlaneInput;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CudaReconstructionInput {
    pub(crate) arenas: Arc<[CudaArenaInput]>,
    pub(crate) planes: Arc<[CudaPlaneInput]>,
    pub(crate) low_len: usize,
    pub(crate) sample_len: usize,
}

#[derive(Clone, Copy)]
struct PlaneWindow {
    origin_x: u32,
    origin_y: u32,
    macroblocks_x: u32,
    macroblocks_y: u32,
    sample_width: u32,
    sample_height: u32,
}

impl CudaDecodePlan {
    /// Validate a metadata-only coefficient and output contract.
    pub fn new(
        reconstructed_coefficients: u64,
        coefficient_bytes: usize,
        output: SurfaceLayout,
    ) -> Result<Self, CudaError> {
        if reconstructed_coefficients == 0 {
            return Err(invalid("reconstructed coefficient count must be nonzero"));
        }
        let expected = reconstructed_coefficients
            .checked_mul(4)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| invalid("coefficient arena byte length overflows usize"))?;
        if coefficient_bytes != expected {
            return Err(invalid(
                "coefficient arena must contain one i32 per coefficient",
            ));
        }
        validate_output(&output)?;
        Ok(Self {
            reconstructed_coefficients,
            coefficient_bytes,
            output,
            info: None,
            output_region: None,
            output_policy: None,
            requested_backend: BackendRequest::Cuda,
            reconstruction: None,
        })
    }

    /// Build an executable CUDA plan from the shared prepared reconstruction ABI.
    #[allow(clippy::too_many_arguments)]
    pub fn from_prepared(
        primary: Arc<CoefficientArena>,
        separate_alpha: Option<(Arc<CoefficientArena>, PreparedPlan)>,
        prepared: &PreparedPlan,
        output_policy: OutputFormatRequest,
        output: SurfaceLayout,
        coded_origin: [u32; 2],
        requested_backend: BackendRequest,
    ) -> Result<Self, CudaError> {
        validate_plan_contract(prepared, output_policy, &output, coded_origin)?;
        let primary_components = usize::from(
            prepared
                .info
                .primary
                .color_format
                .component_count()
                .ok_or_else(|| invalid("primary component count is zero"))?,
        );
        let integrated_alpha = prepared.info.alpha_mode == AlphaMode::Integrated;
        let expected_primary_planes = primary_components + usize::from(integrated_alpha);
        let arenas = build_arenas(primary, separate_alpha, prepared, expected_primary_planes)?;
        let (planes, low_len, sample_len) = build_planes(
            &arenas,
            prepared,
            primary_components,
            expected_primary_planes,
            integrated_alpha,
        )?;
        let reconstructed_coefficients = reconstruction_work(&planes)?;
        if reconstructed_coefficients
            != prepared
                .reconstructed_coefficients()
                .map_err(|_| invalid("prepared reconstruction work count is invalid"))?
        {
            return Err(invalid(
                "coefficient planes differ from prepared reconstruction work",
            ));
        }
        let coefficient_bytes = coefficient_bytes(&arenas)?;
        let reconstruction = CudaReconstructionInput {
            arenas: arenas.into(),
            planes: planes.into(),
            low_len,
            sample_len,
        };
        let plan = Self {
            reconstructed_coefficients,
            coefficient_bytes,
            output,
            info: Some(prepared.info.clone()),
            output_region: Some(prepared.output_region),
            output_policy: Some(output_policy),
            requested_backend,
            reconstruction: Some(reconstruction),
        };
        plan.validate_device_contract()?;
        Ok(plan)
    }

    /// Planned reconstruction work.
    #[must_use]
    pub const fn reconstructed_coefficients(&self) -> u64 {
        self.reconstructed_coefficients
    }

    /// Bytes in the CPU-produced coefficient input.
    #[must_use]
    pub const fn coefficient_bytes(&self) -> usize {
        self.coefficient_bytes
    }

    /// Planned output surface.
    #[must_use]
    pub const fn output(&self) -> &SurfaceLayout {
        &self.output
    }

    /// Parsed image metadata for an executable plan.
    #[must_use]
    pub const fn info(&self) -> Option<&ImageInfo> {
        self.info.as_ref()
    }

    /// Full-resolution requested output region.
    #[must_use]
    pub const fn output_region(&self) -> Option<Rect> {
        self.output_region
    }

    /// Exact conversion and packing policy.
    #[must_use]
    pub const fn output_policy(&self) -> Option<OutputFormatRequest> {
        self.output_policy
    }

    /// Backend policy carried by the original request.
    #[must_use]
    pub const fn requested_backend(&self) -> BackendRequest {
        self.requested_backend
    }

    /// Whether this plan contains coefficient storage and complete geometry.
    #[must_use]
    pub const fn is_executable(&self) -> bool {
        self.reconstruction.is_some()
    }

    pub(crate) fn reconstruction(&self) -> Result<&CudaReconstructionInput, CudaError> {
        self.reconstruction
            .as_ref()
            .ok_or_else(|| invalid("metadata-only plan has no coefficient arena"))
    }

    pub(crate) fn scratch_bytes(&self) -> Result<usize, CudaError> {
        let input = self.reconstruction()?;
        input
            .low_len
            .checked_add(input.sample_len)
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<i32>()))
            .and_then(|bytes| bytes.checked_add(core::mem::size_of::<u32>()))
            .ok_or_else(|| invalid("CUDA scratch byte count overflows usize"))
    }

    pub(crate) fn decode_report(&self, host_readback: bool) -> DecodeReport {
        const CPU_STAGES: [DecodeStage; 5] = [
            DecodeStage::Parse,
            DecodeStage::EntropyDecode,
            DecodeStage::InverseScan,
            DecodeStage::CoefficientRemap,
            DecodeStage::DcLowPassPrediction,
        ];
        const CUDA_STAGES: [DecodeStage; 8] = [
            DecodeStage::DequantizeAndFirstInverseTransform,
            DecodeStage::FirstOverlap,
            DecodeStage::HighPassPrediction,
            DecodeStage::SecondInverseTransform,
            DecodeStage::SecondOverlap,
            DecodeStage::ChromaReconstruction,
            DecodeStage::ColorAndAlphaConversion,
            DecodeStage::CropClipAndPack,
        ];
        let mut stages = Vec::with_capacity(13 + usize::from(host_readback));
        stages.extend(CPU_STAGES.into_iter().map(|stage| StageReport {
            stage,
            executor: StageExecutor::CpuScalar,
        }));
        stages.extend(CUDA_STAGES.into_iter().map(|stage| StageReport {
            stage,
            executor: StageExecutor::Cuda,
        }));
        if host_readback {
            stages.push(StageReport {
                stage: DecodeStage::HostReadback,
                executor: StageExecutor::CpuScalar,
            });
        }
        DecodeReport {
            requested: self.requested_backend,
            selected: BackendKind::Cuda,
            fallback: None,
            stages,
        }
    }

    fn validate_device_contract(&self) -> Result<(), CudaError> {
        let input = self.reconstruction()?;
        for &plane in input.planes.iter() {
            let arena = reconstruction_arena(input, plane.arena_index)?;
            crate::abi::JxrPlaneAbi::from_plan(plane)?;
            if arena.overlap == OverlapMode::Two {
                crate::overlap_plan::first_overlap_schedule(
                    plane,
                    arena.hard_tiles,
                    &arena.tile_column_widths,
                    &arena.tile_row_heights,
                )?;
            }
            if arena.overlap != OverlapMode::None {
                crate::overlap_plan::second_overlap_schedule(
                    plane,
                    arena.hard_tiles,
                    &arena.tile_column_widths,
                    &arena.tile_row_heights,
                )?;
            }
        }
        self.validate_output_sample_windows(input)?;
        crate::output_plan::build_output_dispatch(self)?;
        Ok(())
    }

    fn validate_output_sample_windows(
        &self,
        input: &CudaReconstructionInput,
    ) -> Result<(), CudaError> {
        let policy = self
            .output_policy
            .ok_or_else(|| invalid("executable plan omits its output policy"))?;
        let info = self
            .info
            .as_ref()
            .ok_or_else(|| invalid("executable plan omits image metadata"))?;
        let crop_right = policy
            .crop
            .x
            .checked_add(policy.crop.width)
            .ok_or_else(|| invalid("output crop right edge overflows u32"))?;
        let crop_bottom = policy
            .crop
            .y
            .checked_add(policy.crop.height)
            .ok_or_else(|| invalid("output crop bottom edge overflows u32"))?;
        for (index, plane) in input.planes.iter().enumerate() {
            let chroma_sampling = (!plane.alpha && matches!(index, 1 | 2)).then_some(
                match info.primary.color_format {
                    ColorFormat::Yuv(sampling) => sampling,
                    _ => ChromaSampling::Cs444,
                },
            );
            let x_divisor = if matches!(
                chroma_sampling,
                Some(ChromaSampling::Cs420 | ChromaSampling::Cs422)
            ) {
                2
            } else {
                1
            };
            let y_divisor = if chroma_sampling == Some(ChromaSampling::Cs420) {
                2
            } else {
                1
            };
            let required_left = policy.crop.x / x_divisor;
            let required_top = policy.crop.y / y_divisor;
            let required_right = crop_right.saturating_sub(1) / x_divisor + 1;
            let required_bottom = crop_bottom.saturating_sub(1) / y_divisor + 1;
            let plane_right = plane
                .sample_origin_x
                .checked_add(plane.sample_width)
                .ok_or_else(|| invalid("CUDA sample plane right edge overflows u32"))?;
            let plane_bottom = plane
                .sample_origin_y
                .checked_add(plane.sample_height)
                .ok_or_else(|| invalid("CUDA sample plane bottom edge overflows u32"))?;
            if required_left < plane.sample_origin_x
                || required_top < plane.sample_origin_y
                || required_right > plane_right
                || required_bottom > plane_bottom
            {
                return Err(invalid("output crop is outside a CUDA sample plane"));
            }
            if chroma_sampling.is_some()
                && (plane.sample_origin_x > i32::MAX as u32
                    || plane.sample_origin_y > i32::MAX as u32
                    || plane_right > i32::MAX as u32
                    || plane_bottom > i32::MAX as u32)
            {
                return Err(invalid(
                    "chroma sample coordinates exceed CUDA signed indexing",
                ));
            }
        }
        Ok(())
    }
}

fn build_arenas(
    primary: Arc<CoefficientArena>,
    separate_alpha: Option<(Arc<CoefficientArena>, PreparedPlan)>,
    prepared: &PreparedPlan,
    expected_primary_planes: usize,
) -> Result<Vec<CudaArenaInput>, CudaError> {
    validate_arena(&primary)?;
    if primary.planes.len() != expected_primary_planes {
        return Err(invalid(
            "primary coefficient arena component count does not match the image",
        ));
    }
    let mut arenas = vec![arena_input(primary, prepared)];
    if let Some((coefficients, alpha_plan)) = separate_alpha {
        validate_arena(&coefficients)?;
        if coefficients.planes.len() != 1 || alpha_plan.info.alpha_mode != AlphaMode::None {
            return Err(invalid(
                "separate alpha arena is not one independent luma plane",
            ));
        }
        if alpha_plan.output_region != prepared.output_region {
            return Err(invalid("separate alpha output region differs from primary"));
        }
        arenas.push(arena_input(coefficients, &alpha_plan));
    }
    Ok(arenas)
}

fn arena_input(source: Arc<CoefficientArena>, plan: &PreparedPlan) -> CudaArenaInput {
    CudaArenaInput {
        source,
        overlap: plan.primary.overlap,
        hard_tiles: plan.info.tiles.hard_tiles,
        tile_column_widths: plan.info.tiles.column_widths.clone().into(),
        tile_row_heights: plan.info.tiles.row_heights.clone().into(),
    }
}

pub(crate) fn reconstruction_arena(
    input: &CudaReconstructionInput,
    index: u32,
) -> Result<&CudaArenaInput, CudaError> {
    input
        .arenas
        .get(usize::try_from(index).map_err(|_| invalid("arena index does not fit usize"))?)
        .ok_or_else(|| invalid("arena index is out of range"))
}

fn build_planes(
    arenas: &[CudaArenaInput],
    prepared: &PreparedPlan,
    primary_components: usize,
    expected_primary_planes: usize,
    integrated_alpha: bool,
) -> Result<(Vec<CudaPlaneInput>, usize, usize), CudaError> {
    let mut planes = Vec::with_capacity(expected_primary_planes + usize::from(arenas.len() > 1));
    let mut low_len = 0;
    let mut sample_len = 0;
    for plane_index in 0..expected_primary_planes {
        let alpha = integrated_alpha && plane_index == primary_components;
        let scale_chroma = !alpha
            && plane_index != 0
            && prepared.info.primary.scaled
            && matches!(
                prepared.info.primary.color_format,
                jxr_core::ColorFormat::Yuv(_)
            );
        planes.push(build_plane(
            &arenas[0],
            plane_index,
            0,
            scale_chroma,
            alpha,
            &mut low_len,
            &mut sample_len,
        )?);
    }
    if arenas.len() > 1 {
        planes.push(build_plane(
            &arenas[1],
            0,
            1,
            false,
            true,
            &mut low_len,
            &mut sample_len,
        )?);
    }
    Ok((planes, low_len, sample_len))
}

#[allow(clippy::too_many_arguments)]
fn build_plane(
    arena: &CudaArenaInput,
    plane_index: usize,
    arena_index: u32,
    scale_after_first_transform: bool,
    alpha: bool,
    low_len: &mut usize,
    sample_len: &mut usize,
) -> Result<CudaPlaneInput, CudaError> {
    let source = *arena
        .source
        .planes
        .get(plane_index)
        .ok_or_else(|| invalid("coefficient plane index is absent"))?;
    if !matches!(
        (source.block_columns, source.block_rows),
        (4 | 2, 4) | (2, 2)
    ) {
        return Err(invalid("unsupported component macroblock geometry"));
    }
    let window = plane_window(&arena.source.macroblocks, source)?;
    let (low_offset, sample_offset) = reserve_plane_regions(source, window, low_len, sample_len)?;
    let block_columns = u32::from(source.block_columns);
    let block_rows = u32::from(source.block_rows);
    Ok(CudaPlaneInput {
        arena_index,
        macroblock_offset: source.macroblock_offset,
        macroblock_count: source.macroblock_count,
        block_columns: source.block_columns,
        block_rows: source.block_rows,
        macroblock_origin_x: window.origin_x,
        macroblock_origin_y: window.origin_y,
        macroblocks_x: window.macroblocks_x,
        macroblocks_y: window.macroblocks_y,
        sample_origin_x: window
            .origin_x
            .checked_mul(block_columns * 4)
            .ok_or_else(|| invalid("component sample x origin overflows u32"))?,
        sample_origin_y: window
            .origin_y
            .checked_mul(block_rows * 4)
            .ok_or_else(|| invalid("component sample y origin overflows u32"))?,
        sample_width: window.sample_width,
        sample_height: window.sample_height,
        low_offset,
        sample_offset,
        scale_after_first_transform,
        alpha,
    })
}

fn plane_window(
    macroblocks: &MacroblockMetadata,
    source: CoefficientPlane,
) -> Result<PlaneWindow, CudaError> {
    let end = source
        .macroblock_offset
        .checked_add(source.macroblock_count)
        .ok_or_else(|| invalid("macroblock metadata range overflows usize"))?;
    let xs = macroblocks
        .coded_x
        .get(source.macroblock_offset..end)
        .ok_or_else(|| invalid("coded x metadata range is invalid"))?;
    let ys = macroblocks
        .coded_y
        .get(source.macroblock_offset..end)
        .ok_or_else(|| invalid("coded y metadata range is invalid"))?;
    let (&origin_x, &origin_y) = xs
        .iter()
        .min()
        .zip(ys.iter().min())
        .ok_or_else(|| invalid("coefficient plane has no macroblocks"))?;
    let right = xs
        .iter()
        .max()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| invalid("macroblock x extent overflows u32"))?;
    let bottom = ys
        .iter()
        .max()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| invalid("macroblock y extent overflows u32"))?;
    let macroblocks_x = right - origin_x;
    let macroblocks_y = bottom - origin_y;
    let expected = usize::try_from(macroblocks_x)
        .ok()
        .and_then(|width| {
            usize::try_from(macroblocks_y)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| invalid("macroblock window size overflows usize"))?;
    if expected != source.macroblock_count {
        return Err(invalid(
            "coefficient plane reconstruction window is not rectangular",
        ));
    }
    let width = usize::try_from(macroblocks_x)
        .map_err(|_| invalid("macroblock window width does not fit usize"))?;
    let mut occupied = vec![false; expected];
    for (&x, &y) in xs.iter().zip(ys) {
        let local_x = usize::try_from(x - origin_x)
            .map_err(|_| invalid("macroblock x coordinate does not fit usize"))?;
        let local_y = usize::try_from(y - origin_y)
            .map_err(|_| invalid("macroblock y coordinate does not fit usize"))?;
        let index = local_y
            .checked_mul(width)
            .and_then(|row| row.checked_add(local_x))
            .filter(|&index| index < occupied.len())
            .ok_or_else(|| invalid("macroblock coordinate exceeds its reconstruction window"))?;
        if core::mem::replace(&mut occupied[index], true) {
            return Err(invalid("coefficient plane repeats a macroblock coordinate"));
        }
    }
    let sample_width = macroblocks_x
        .checked_mul(u32::from(source.block_columns))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| invalid("component sample width overflows u32"))?;
    let sample_height = macroblocks_y
        .checked_mul(u32::from(source.block_rows))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| invalid("component sample height overflows u32"))?;
    Ok(PlaneWindow {
        origin_x,
        origin_y,
        macroblocks_x,
        macroblocks_y,
        sample_width,
        sample_height,
    })
}

fn reserve_plane_regions(
    source: CoefficientPlane,
    window: PlaneWindow,
    low_len: &mut usize,
    sample_len: &mut usize,
) -> Result<(usize, usize), CudaError> {
    let low_offset = align_i32(*low_len)?;
    let macroblocks = usize::try_from(window.macroblocks_x)
        .ok()
        .and_then(|width| {
            usize::try_from(window.macroblocks_y)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| invalid("macroblock window size overflows usize"))?;
    let low_count = macroblocks
        .checked_mul(usize::from(source.block_columns) * usize::from(source.block_rows))
        .ok_or_else(|| invalid("low-pass plane length overflows usize"))?;
    *low_len = low_offset
        .checked_add(low_count)
        .ok_or_else(|| invalid("low-pass arena length overflows usize"))?;
    let sample_offset = align_i32(*sample_len)?;
    let sample_count = usize::try_from(window.sample_width)
        .ok()
        .and_then(|width| {
            usize::try_from(window.sample_height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| invalid("sample plane length overflows usize"))?;
    *sample_len = sample_offset
        .checked_add(sample_count)
        .ok_or_else(|| invalid("sample arena length overflows usize"))?;
    Ok((low_offset, sample_offset))
}

fn reconstruction_work(planes: &[CudaPlaneInput]) -> Result<u64, CudaError> {
    planes.iter().try_fold(0_u64, |total, plane| {
        let per_macroblock = u64::from(plane.block_columns)
            .checked_mul(u64::from(plane.block_rows))
            .and_then(|blocks| blocks.checked_mul(16))
            .ok_or_else(|| invalid("plane work count overflows u64"))?;
        let plane_work = u64::try_from(plane.macroblock_count)
            .ok()
            .and_then(|count| count.checked_mul(per_macroblock))
            .ok_or_else(|| invalid("reconstructed coefficient count overflows u64"))?;
        total
            .checked_add(plane_work)
            .ok_or_else(|| invalid("reconstructed coefficient count overflows u64"))
    })
}

fn validate_plan_contract(
    prepared: &PreparedPlan,
    output_policy: OutputFormatRequest,
    output: &SurfaceLayout,
    coded_origin: [u32; 2],
) -> Result<(), CudaError> {
    if prepared.scale != jxr_core::DecodeScale::Full {
        return Err(CudaError::UnsupportedOutputFormat {
            reason: "native reduced reconstruction is CPU-only",
        });
    }
    validate_output(output)?;
    if output.width != prepared.output_region.w
        || output.height != prepared.output_region.h
        || output.format != output_policy.pixel_format
        || output_policy.crop.width != prepared.output_region.w
        || output_policy.crop.height != prepared.output_region.h
    {
        return Err(invalid("output layout and prepared request differ"));
    }
    let expected_crop_x = coded_origin[0]
        .checked_add(prepared.output_region.x)
        .ok_or_else(|| invalid("output crop x overflows u32"))?;
    let expected_crop_y = coded_origin[1]
        .checked_add(prepared.output_region.y)
        .ok_or_else(|| invalid("output crop y overflows u32"))?;
    if output_policy.crop.x != expected_crop_x || output_policy.crop.y != expected_crop_y {
        return Err(invalid(
            "output crop origin differs from the prepared image",
        ));
    }
    Ok(())
}

fn coefficient_bytes(arenas: &[CudaArenaInput]) -> Result<usize, CudaError> {
    arenas.iter().try_fold(0_usize, |total, arena| {
        arena
            .source
            .coefficients
            .len()
            .checked_mul(core::mem::size_of::<i32>())
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| invalid("coefficient byte length overflows usize"))
    })
}

fn align_i32(value: usize) -> Result<usize, CudaError> {
    value
        .checked_add(INTERNAL_ALIGNMENT_I32 - 1)
        .map(|value| value & !(INTERNAL_ALIGNMENT_I32 - 1))
        .ok_or_else(|| invalid("internal plane alignment overflows usize"))
}

fn validate_arena(arena: &CoefficientArena) -> Result<(), CudaError> {
    arena
        .validate()
        .map_err(|_| invalid("coefficient arena does not satisfy the jxr-core contract"))?;
    if arena
        .macroblocks
        .hp_predictions
        .contains(&PredictionMode::FromTopLeft)
    {
        return Err(invalid("top-left high-pass prediction is invalid"));
    }
    for plane in &arena.planes {
        let block_count = usize::from(plane.block_columns)
            .checked_mul(usize::from(plane.block_rows))
            .ok_or_else(|| invalid("component block count overflows usize"))?;
        let end = plane
            .macroblock_offset
            .checked_add(plane.macroblock_count)
            .ok_or_else(|| invalid("macroblock plane range overflows usize"))?;
        for index in plane.macroblock_offset..end {
            let required = match arena.macroblocks.bands[index] {
                jxr_core::BandPresence::DcOnly => 1,
                jxr_core::BandPresence::NoHighPass => block_count,
                jxr_core::BandPresence::NoFlexbits | jxr_core::BandPresence::All => block_count
                    .checked_mul(16)
                    .ok_or_else(|| invalid("macroblock coefficient span overflows usize"))?,
            };
            let offset = usize::try_from(arena.macroblocks.coefficient_offsets[index])
                .map_err(|_| invalid("coefficient offset does not fit usize"))?;
            let coefficient_end = offset
                .checked_add(required)
                .ok_or_else(|| invalid("macroblock coefficient span overflows usize"))?;
            if coefficient_end > arena.coefficients.len() {
                return Err(invalid(
                    "macroblock coefficient span exceeds the uploaded arena",
                ));
            }
        }
    }
    Ok(())
}

fn validate_output(output: &SurfaceLayout) -> Result<(), CudaError> {
    output
        .validate()
        .map_err(|_| invalid("output surface does not satisfy the jxr-core contract"))?;
    u32::try_from(output.byte_len)
        .map_err(|_| invalid("output surface byte length exceeds the CUDA ABI"))?;
    Ok(())
}

const fn invalid(reason: &'static str) -> CudaError {
    CudaError::InvalidPlan { reason }
}

#[cfg(test)]
mod tests;
