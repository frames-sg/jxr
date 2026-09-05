// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use jxr_core::{BackendKind, DecodeReport, DecodeStage, StageExecutor, StageReport};

use std::sync::Arc;

use jxr_core::{
    AlphaMode, BackendRequest, CoefficientArena, CoefficientPlane, ImageInfo, MacroblockMetadata,
    OutputFormatRequest, OverlapMode, PreparedPlan, Rect, SurfaceLayout,
};

use crate::MetalError;

mod input;
use input::{build_arenas, build_planes};

const INTERNAL_ALIGNMENT_I32: usize = 64;

/// Device-neutral data required before Metal reconstruction can be submitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDecodePlan {
    reconstructed_coefficients: u64,
    coefficient_bytes: usize,
    output: SurfaceLayout,
    info: Option<ImageInfo>,
    output_region: Option<Rect>,
    output_policy: Option<OutputFormatRequest>,
    requested_backend: BackendRequest,
    reconstruction: Option<MetalReconstructionInput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalArenaInput {
    pub(crate) source: MetalCoefficientSource,
}

#[derive(Debug, Clone)]
pub(crate) enum MetalCoefficientSource {
    Cpu(Arc<CoefficientArena>),
    Shared(Arc<crate::MetalCoefficientArena>),
}

impl PartialEq for MetalCoefficientSource {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu(left), Self::Cpu(right)) => left == right,
            (Self::Shared(left), Self::Shared(right)) => Arc::ptr_eq(left, right),
            (Self::Cpu(_), Self::Shared(_)) | (Self::Shared(_), Self::Cpu(_)) => false,
        }
    }
}

impl Eq for MetalCoefficientSource {}

impl MetalArenaInput {
    pub(crate) fn coefficient_count(&self) -> usize {
        match &self.source {
            MetalCoefficientSource::Cpu(arena) => arena.coefficients.len(),
            MetalCoefficientSource::Shared(arena) => arena.descriptor().coefficient_count,
        }
    }

    pub(crate) fn macroblocks(&self) -> &MacroblockMetadata {
        match &self.source {
            MetalCoefficientSource::Cpu(arena) => &arena.macroblocks,
            MetalCoefficientSource::Shared(arena) => &arena.descriptor().macroblocks,
        }
    }

    pub(crate) fn planes(&self) -> &[CoefficientPlane] {
        match &self.source {
            MetalCoefficientSource::Cpu(arena) => &arena.planes,
            MetalCoefficientSource::Shared(arena) => &arena.descriptor().planes,
        }
    }
}

pub(crate) use jxr_core::device_plan::ReconstructionPlane as MetalPlaneInput;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalReconstructionInput {
    pub(crate) arenas: Arc<[MetalArenaInput]>,
    pub(crate) planes: Arc<[MetalPlaneInput]>,
    pub(crate) overlap: OverlapMode,
    pub(crate) hard_tiles: bool,
    pub(crate) tile_column_widths: Arc<[u32]>,
    pub(crate) tile_row_heights: Arc<[u32]>,
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

impl MetalDecodePlan {
    /// Validate a metadata-only coefficient and output contract.
    pub fn new(
        reconstructed_coefficients: u64,
        coefficient_bytes: usize,
        output: SurfaceLayout,
    ) -> Result<Self, MetalError> {
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
            requested_backend: BackendRequest::Metal,
            reconstruction: None,
        })
    }

    /// Build an executable multi-plane Metal plan from CPU-owned entropy output.
    #[allow(clippy::too_many_arguments)]
    pub fn from_prepared(
        primary: Arc<CoefficientArena>,
        separate_alpha: Option<(Arc<CoefficientArena>, PreparedPlan)>,
        prepared: &PreparedPlan,
        output_policy: OutputFormatRequest,
        output: SurfaceLayout,
        coded_origin: [u32; 2],
        requested_backend: BackendRequest,
    ) -> Result<Self, MetalError> {
        validate_plan_contract(prepared, output_policy, &output, coded_origin)?;
        let primary_components = prepared
            .info
            .primary
            .color_format
            .component_count()
            .ok_or_else(|| invalid("primary component count is zero"))?;
        let primary_components = usize::from(primary_components);
        let integrated_alpha = prepared.info.alpha_mode == AlphaMode::Integrated;
        let expected_primary_planes = primary_components + usize::from(integrated_alpha);
        let arenas = build_arenas(primary, separate_alpha, prepared, expected_primary_planes)?;
        Self::from_arenas(
            arenas,
            prepared,
            output_policy,
            output,
            requested_backend,
            primary_components,
            expected_primary_planes,
            integrated_alpha,
        )
    }

    /// Build an executable plan whose primary entropy output already occupies shared Metal storage.
    #[allow(clippy::too_many_arguments)]
    pub fn from_staged_primary(
        primary: Arc<crate::MetalCoefficientArena>,
        prepared: &PreparedPlan,
        output_policy: OutputFormatRequest,
        output: SurfaceLayout,
        coded_origin: [u32; 2],
        requested_backend: BackendRequest,
    ) -> Result<Self, MetalError> {
        validate_plan_contract(prepared, output_policy, &output, coded_origin)?;
        if prepared.info.alpha_mode == AlphaMode::Separate {
            return Err(invalid(
                "direct shared staging does not yet combine separate alpha",
            ));
        }
        primary
            .descriptor()
            .validate()
            .map_err(|_| invalid("shared primary coefficient descriptor is invalid"))?;
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
        if primary.descriptor().planes.len() != expected_primary_planes {
            return Err(invalid(
                "shared primary component count does not match the image",
            ));
        }
        Self::from_arenas(
            vec![MetalArenaInput {
                source: MetalCoefficientSource::Shared(primary),
            }],
            prepared,
            output_policy,
            output,
            requested_backend,
            primary_components,
            expected_primary_planes,
            integrated_alpha,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_arenas(
        arenas: Vec<MetalArenaInput>,
        prepared: &PreparedPlan,
        output_policy: OutputFormatRequest,
        output: SurfaceLayout,
        requested_backend: BackendRequest,
        primary_components: usize,
        expected_primary_planes: usize,
        integrated_alpha: bool,
    ) -> Result<Self, MetalError> {
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
        let reconstruction = MetalReconstructionInput {
            arenas: arenas.into(),
            planes: planes.into(),
            overlap: prepared.primary.overlap,
            hard_tiles: prepared.info.tiles.hard_tiles,
            tile_column_widths: prepared.info.tiles.column_widths.clone().into(),
            tile_row_heights: prepared.info.tiles.row_heights.clone().into(),
            low_len,
            sample_len,
        };
        Ok(Self {
            reconstructed_coefficients,
            coefficient_bytes,
            output,
            info: Some(prepared.info.clone()),
            output_region: Some(prepared.output_region),
            output_policy: Some(output_policy),
            requested_backend,
            reconstruction: Some(reconstruction),
        })
    }

    #[must_use]
    pub const fn reconstructed_coefficients(&self) -> u64 {
        self.reconstructed_coefficients
    }

    #[must_use]
    pub const fn coefficient_bytes(&self) -> usize {
        self.coefficient_bytes
    }

    #[must_use]
    pub const fn output(&self) -> &SurfaceLayout {
        &self.output
    }

    #[must_use]
    pub const fn info(&self) -> Option<&ImageInfo> {
        self.info.as_ref()
    }

    #[must_use]
    pub const fn output_region(&self) -> Option<Rect> {
        self.output_region
    }

    #[must_use]
    pub const fn output_policy(&self) -> Option<OutputFormatRequest> {
        self.output_policy
    }

    #[must_use]
    pub const fn requested_backend(&self) -> BackendRequest {
        self.requested_backend
    }

    #[must_use]
    pub const fn is_executable(&self) -> bool {
        self.reconstruction.is_some()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn reconstruction(&self) -> Result<&MetalReconstructionInput, MetalError> {
        self.reconstruction
            .as_ref()
            .ok_or_else(|| invalid("metadata-only plan has no coefficient arena"))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn scratch_bytes(&self) -> Result<usize, MetalError> {
        let input = self.reconstruction()?;
        input
            .low_len
            .checked_add(input.sample_len)
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<i32>()))
            .and_then(|bytes| bytes.checked_add(core::mem::size_of::<u32>()))
            .ok_or_else(|| invalid("Metal scratch byte count overflows usize"))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn decode_report(&self, host_readback: bool) -> DecodeReport {
        const CPU_STAGES: [DecodeStage; 5] = [
            DecodeStage::Parse,
            DecodeStage::EntropyDecode,
            DecodeStage::InverseScan,
            DecodeStage::CoefficientRemap,
            DecodeStage::DcLowPassPrediction,
        ];
        const METAL_STAGES: [DecodeStage; 8] = [
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
        stages.extend(METAL_STAGES.into_iter().map(|stage| StageReport {
            stage,
            executor: StageExecutor::Metal,
        }));
        if host_readback {
            stages.push(StageReport {
                stage: DecodeStage::HostReadback,
                executor: StageExecutor::CpuScalar,
            });
        }
        DecodeReport {
            requested: self.requested_backend,
            selected: BackendKind::Metal,
            fallback: None,
            stages,
        }
    }
}

fn reconstruction_work(planes: &[MetalPlaneInput]) -> Result<u64, MetalError> {
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
) -> Result<(), MetalError> {
    validate_decode_scale(prepared.scale)?;
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

fn validate_decode_scale(scale: jxr_core::DecodeScale) -> Result<(), MetalError> {
    if scale == jxr_core::DecodeScale::Full {
        Ok(())
    } else {
        Err(MetalError::UnsupportedOutputFormat {
            reason: "native reduced reconstruction is CPU-only",
        })
    }
}

fn coefficient_bytes(arenas: &[MetalArenaInput]) -> Result<usize, MetalError> {
    arenas.iter().try_fold(0_usize, |total, arena| {
        arena
            .coefficient_count()
            .checked_mul(core::mem::size_of::<i32>())
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| invalid("coefficient byte length overflows usize"))
    })
}

#[allow(clippy::too_many_arguments)]
fn build_plane(
    arena: &MetalArenaInput,
    plane_index: usize,
    arena_index: u32,
    scale_after_first_transform: bool,
    alpha: bool,
    low_len: &mut usize,
    sample_len: &mut usize,
) -> Result<MetalPlaneInput, MetalError> {
    let source = *arena
        .planes()
        .get(plane_index)
        .ok_or_else(|| invalid("coefficient plane index is absent"))?;
    if !matches!(
        (source.block_columns, source.block_rows),
        (4 | 2, 4) | (2, 2)
    ) {
        return Err(invalid("unsupported component macroblock geometry"));
    }
    let window = plane_window(arena.macroblocks(), source)?;
    let (low_offset, sample_offset) = reserve_plane_regions(source, window, low_len, sample_len)?;
    let block_columns = u32::from(source.block_columns);
    let block_rows = u32::from(source.block_rows);
    Ok(MetalPlaneInput {
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
) -> Result<PlaneWindow, MetalError> {
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
    let block_columns = u32::from(source.block_columns);
    let block_rows = u32::from(source.block_rows);
    let sample_width = macroblocks_x
        .checked_mul(block_columns)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| invalid("component sample width overflows u32"))?;
    let sample_height = macroblocks_y
        .checked_mul(block_rows)
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
) -> Result<(usize, usize), MetalError> {
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

fn align_i32(value: usize) -> Result<usize, MetalError> {
    value
        .checked_add(INTERNAL_ALIGNMENT_I32 - 1)
        .map(|value| value & !(INTERNAL_ALIGNMENT_I32 - 1))
        .ok_or_else(|| invalid("internal plane alignment overflows usize"))
}

fn validate_arena(arena: &CoefficientArena) -> Result<(), MetalError> {
    arena
        .validate()
        .map_err(|_| invalid("coefficient arena does not satisfy the jxr-core contract"))
}

fn validate_output(output: &SurfaceLayout) -> Result<(), MetalError> {
    output
        .validate()
        .map_err(|_| invalid("output surface does not satisfy the jxr-core contract"))
}

const fn invalid(reason: &'static str) -> MetalError {
    MetalError::InvalidPlan { reason }
}

#[cfg(test)]
mod tests {
    use jxr_core::{
        BandPresence, ChannelLayout, CoefficientPlane, MacroblockMetadata, PixelFormat,
        PredictionMode, QuantizerSet, SurfacePlaneLayout, TileEdgeFlags,
    };

    use super::*;

    fn output() -> SurfaceLayout {
        SurfaceLayout {
            width: 8,
            height: 8,
            format: PixelFormat::U8(ChannelLayout::Rgba),
            planes: vec![SurfacePlaneLayout {
                byte_offset: 0,
                row_stride_bytes: 32,
                width: 8,
                height: 8,
                channels: 4,
            }],
            byte_len: 256,
            required_alignment: 4,
        }
    }

    #[test]
    fn validates_coefficient_arena_exactly() {
        assert!(MetalDecodePlan::new(64, 256, output()).is_ok());
        assert!(MetalDecodePlan::new(64, 255, output()).is_err());
    }

    #[test]
    fn rejects_invalid_shared_surface() {
        let mut invalid = output();
        invalid.planes[0].row_stride_bytes = 31;
        assert!(MetalDecodePlan::new(64, 256, invalid).is_err());
    }

    #[test]
    fn rejects_native_reduction_before_metal_resource_planning() {
        assert!(validate_decode_scale(jxr_core::DecodeScale::Full).is_ok());
        assert!(validate_decode_scale(jxr_core::DecodeScale::Quarter).is_err());
        assert!(validate_decode_scale(jxr_core::DecodeScale::Sixteenth).is_err());
    }

    #[test]
    fn component_geometry_uses_sparse_window_and_aligned_shared_arenas() {
        let arena = CoefficientArena {
            coefficients: vec![0],
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
                coded_x: vec![3],
                coded_y: vec![2],
                output_x: vec![0],
                output_y: vec![0],
            },
            planes: vec![CoefficientPlane {
                coefficient_offset: 0,
                coefficient_count: 1,
                macroblock_offset: 0,
                macroblock_count: 1,
                block_columns: 2,
                block_rows: 2,
            }],
        };
        let arena = MetalArenaInput {
            source: MetalCoefficientSource::Cpu(Arc::new(arena)),
        };
        let mut low_len = 0;
        let mut sample_len = 0;
        let plane = build_plane(&arena, 0, 0, true, false, &mut low_len, &mut sample_len).unwrap();
        assert_eq!((plane.sample_origin_x, plane.sample_origin_y), (24, 16));
        assert_eq!((plane.sample_width, plane.sample_height), (8, 8));
        assert_eq!(low_len, 4);
        assert_eq!(sample_len, 64);
    }
}
