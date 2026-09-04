//! Ordered scalar execution of the T.832 clause 9.9 reconstruction stages.

use jxr_core::{DecodeScale, OverlapMode, QuantizerSet};
use jxr_math::quantization::Quantizer;
use rayon::prelude::*;

use crate::CpuCapabilities;

use super::{
    PlanarSamples, QuantizedMacroblock, ReconstructionConfig, ReconstructionError,
    overlap::apply_overlap,
    simd_dequant::scale_coefficients,
    subsampled_overlap::apply_subsampled_first_overlap,
    transform::{inverse_chroma_420, inverse_chroma_422, inverse_core_transform},
};

/// Retained per-component transform buffers and recyclable output plane.
#[derive(Debug, Default)]
pub(crate) struct ReconstructionPipelineWorkspace {
    dc_low_pass: Vec<[i32; 16]>,
    high_pass: Vec<[i32; 256]>,
    low_plane: Vec<i32>,
    output_samples: Vec<i32>,
    reuses: u64,
    output_reuses: u64,
}

#[derive(Clone, Copy)]
struct ReconstructionGeometry {
    macroblocks_x: usize,
    macroblocks_y: usize,
    block_columns: usize,
    block_rows: usize,
}

impl ReconstructionPipelineWorkspace {
    pub(crate) const fn reuses(&self) -> u64 {
        self.reuses.saturating_add(self.output_reuses)
    }

    #[cfg(test)]
    pub(crate) const fn output_reuses(&self) -> u64 {
        self.output_reuses
    }

    pub(crate) fn retained_output_bytes(&self) -> usize {
        self.output_samples
            .capacity()
            .saturating_mul(core::mem::size_of::<i32>())
    }

    pub(crate) fn recycle_samples(&mut self, mut samples: Vec<i32>) {
        samples.clear();
        if samples.capacity() > self.output_samples.capacity() {
            self.output_samples = samples;
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.dc_low_pass
            .capacity()
            .saturating_mul(core::mem::size_of::<[i32; 16]>())
            .saturating_add(
                self.high_pass
                    .capacity()
                    .saturating_mul(core::mem::size_of::<[i32; 256]>()),
            )
            .saturating_add(
                self.low_plane
                    .capacity()
                    .saturating_mul(core::mem::size_of::<i32>()),
            )
            .saturating_add(self.retained_output_bytes())
    }

    fn take_output_samples(&mut self) -> Vec<i32> {
        core::mem::take(&mut self.output_samples)
    }
}

/// Reconstruct one full-resolution component from quantized coefficients.
///
/// The returned samples are signed transform-domain results. Output bias,
/// post-scaling, and colour conversion must be completed before packing.
pub fn reconstruct_luma(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
) -> Result<PlanarSamples, ReconstructionError> {
    reconstruct_luma_impl(
        macroblocks,
        config,
        None,
        &mut ReconstructionPipelineWorkspace::default(),
    )
}

#[cfg(test)]
pub(crate) fn reconstruct_luma_scaled(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    scale: DecodeScale,
) -> Result<PlanarSamples, ReconstructionError> {
    reconstruct_luma_scaled_impl(
        macroblocks,
        config,
        scale,
        None,
        &mut ReconstructionPipelineWorkspace::default(),
    )
}

pub(crate) fn reconstruct_luma_scaled_with_cpu(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    scale: DecodeScale,
    cpu: CpuCapabilities,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<PlanarSamples, ReconstructionError> {
    reconstruct_luma_scaled_impl(macroblocks, config, scale, Some(cpu), workspace)
}

fn reconstruct_luma_scaled_impl(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    scale: DecodeScale,
    cpu: Option<CpuCapabilities>,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<PlanarSamples, ReconstructionError> {
    match scale {
        DecodeScale::Full => reconstruct_luma_impl(macroblocks, config, cpu, workspace),
        DecodeScale::Quarter => reconstruct_quarter(macroblocks, config, workspace),
        DecodeScale::Sixteenth => reconstruct_sixteenth(macroblocks, config, workspace),
    }
}

fn reconstruct_quarter(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<PlanarSamples, ReconstructionError> {
    let (macroblocks_x, macroblocks_y, block_columns, block_rows) =
        validated_geometry(macroblocks, config)?;
    let block_count = block_columns * block_rows;
    let reused = workspace.dc_low_pass.capacity() >= macroblocks.len();
    workspace.dc_low_pass.resize(macroblocks.len(), [0; 16]);
    if macroblocks.len() >= 64 {
        workspace
            .dc_low_pass
            .par_iter_mut()
            .zip(macroblocks.par_iter())
            .try_for_each(|(low, macroblock)| {
                reconstruct_low_into(macroblock, block_count, config, low)
            })?;
    } else {
        for (low, macroblock) in workspace.dc_low_pass.iter_mut().zip(macroblocks) {
            reconstruct_low_into(macroblock, block_count, config, low)?;
        }
    }
    let width =
        macroblocks_x
            .checked_mul(block_columns)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "quarter-resolution width",
            ))?;
    let height =
        macroblocks_y
            .checked_mul(block_rows)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "quarter-resolution height",
            ))?;
    let output_reused = workspace.output_samples.capacity() >= width.saturating_mul(height);
    combine_low_pass_into(
        &workspace.dc_low_pass,
        macroblocks_x,
        macroblocks_y,
        block_columns,
        block_rows,
        &mut workspace.output_samples,
    )?;
    let samples = workspace.take_output_samples();
    workspace.reuses = workspace.reuses.saturating_add(u64::from(reused));
    workspace.output_reuses = workspace
        .output_reuses
        .saturating_add(u64::from(output_reused));
    Ok(PlanarSamples {
        origin_x: config
            .macroblock_origin_x
            .checked_mul(u32::from(config.block_columns))
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "quarter-resolution x origin",
            ))?,
        origin_y: config
            .macroblock_origin_y
            .checked_mul(u32::from(config.block_rows))
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "quarter-resolution y origin",
            ))?,
        width: u32::try_from(width).map_err(|_| {
            ReconstructionError::ArithmeticOverflow("quarter-resolution width conversion")
        })?,
        height: u32::try_from(height).map_err(|_| {
            ReconstructionError::ArithmeticOverflow("quarter-resolution height conversion")
        })?,
        samples,
    })
}

fn reconstruct_sixteenth(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<PlanarSamples, ReconstructionError> {
    let (macroblocks_x, macroblocks_y, _, _) = validated_geometry(macroblocks, config)?;
    let output_reused = workspace.output_samples.capacity() >= macroblocks.len();
    workspace.output_samples.clear();
    for macroblock in macroblocks {
        let quantizer = quantizer(macroblock.quantizers.dc, "DC")?;
        let mut value = scale(quantizer, macroblock.dc_low_pass[0])?;
        if config.scale_after_first_transform {
            value = value
                .checked_mul(2)
                .ok_or(ReconstructionError::ArithmeticOverflow(
                    "sixteenth-resolution chroma normalization",
                ))?;
        }
        workspace.output_samples.push(value >> 4);
    }
    let samples = workspace.take_output_samples();
    workspace.output_reuses = workspace
        .output_reuses
        .saturating_add(u64::from(output_reused));
    Ok(PlanarSamples {
        origin_x: config.macroblock_origin_x,
        origin_y: config.macroblock_origin_y,
        width: u32::try_from(macroblocks_x).map_err(|_| {
            ReconstructionError::ArithmeticOverflow("sixteenth-resolution width conversion")
        })?,
        height: u32::try_from(macroblocks_y).map_err(|_| {
            ReconstructionError::ArithmeticOverflow("sixteenth-resolution height conversion")
        })?,
        samples,
    })
}

fn validated_geometry(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
) -> Result<(usize, usize, usize, usize), ReconstructionError> {
    let macroblocks_x = usize::try_from(config.macroblocks_x)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("macroblock width conversion"))?;
    let macroblocks_y = usize::try_from(config.macroblocks_y)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("macroblock height conversion"))?;
    let expected = macroblocks_x
        .checked_mul(macroblocks_y)
        .ok_or(ReconstructionError::ArithmeticOverflow("macroblock count"))?;
    if macroblocks.len() != expected {
        return Err(ReconstructionError::MacroblockCount {
            expected,
            actual: macroblocks.len(),
        });
    }
    validate_partition(
        &config.tiles.column_boundaries,
        config.macroblocks_x,
        "column",
    )?;
    validate_partition(&config.tiles.row_boundaries, config.macroblocks_y, "row")?;
    let (block_columns, block_rows) = component_block_geometry(config)?;
    Ok((macroblocks_x, macroblocks_y, block_columns, block_rows))
}

fn reconstruct_luma_impl(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    cpu: Option<CpuCapabilities>,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<PlanarSamples, ReconstructionError> {
    let (macroblocks_x, macroblocks_y, block_columns, block_rows) =
        validated_geometry(macroblocks, config)?;
    let expected = macroblocks.len();
    let block_count = block_columns * block_rows;

    let low_plane_length = macroblocks_x
        .checked_mul(block_columns)
        .and_then(|width| width.checked_mul(macroblocks_y))
        .and_then(|length| length.checked_mul(block_rows))
        .ok_or(ReconstructionError::ArithmeticOverflow(
            "low-pass plane size",
        ))?;
    let reused = workspace.dc_low_pass.capacity() >= expected
        && workspace.high_pass.capacity() >= expected
        && workspace.low_plane.capacity() >= low_plane_length;
    workspace.dc_low_pass.resize(expected, [0; 16]);
    workspace.high_pass.resize(expected, [0; 256]);

    if expected >= 64 {
        workspace
            .dc_low_pass
            .par_iter_mut()
            .zip(workspace.high_pass.par_iter_mut())
            .zip(macroblocks.par_iter())
            .try_for_each(|((low, high), macroblock)| {
                reconstruct_first_level_into(macroblock, block_count, config, cpu, low, high)
            })?;
    } else {
        for ((low, high), macroblock) in workspace
            .dc_low_pass
            .iter_mut()
            .zip(&mut workspace.high_pass)
            .zip(macroblocks)
        {
            reconstruct_first_level_into(macroblock, block_count, config, cpu, low, high)?;
        }
    }

    combine_low_pass_into(
        &workspace.dc_low_pass,
        macroblocks_x,
        macroblocks_y,
        block_columns,
        block_rows,
        &mut workspace.low_plane,
    )?;
    if config.overlap == OverlapMode::Two {
        if block_columns == 4 {
            apply_configured_overlap(
                &mut workspace.low_plane,
                macroblocks_x * block_columns,
                macroblocks_y * block_rows,
                config,
                block_columns,
                block_rows,
            )?;
        } else {
            apply_configured_subsampled_overlap(
                &mut workspace.low_plane,
                macroblocks_x * block_columns,
                macroblocks_y * block_rows,
                config,
                block_columns,
                block_rows,
            )?;
        }
    }
    scatter_low_pass(
        &workspace.low_plane,
        &mut workspace.dc_low_pass,
        macroblocks_x,
        macroblocks_y,
        block_columns,
        block_rows,
    );

    let geometry = ReconstructionGeometry {
        macroblocks_x,
        macroblocks_y,
        block_columns,
        block_rows,
    };
    let (result, output_reused) = finish_component(
        &workspace.dc_low_pass,
        &mut workspace.high_pass,
        geometry,
        config,
        &mut workspace.output_samples,
    )?;
    workspace.reuses = workspace.reuses.saturating_add(u64::from(reused));
    workspace.output_reuses = workspace
        .output_reuses
        .saturating_add(u64::from(output_reused));
    Ok(result)
}

fn reconstruct_first_level_into(
    macroblock: &QuantizedMacroblock,
    block_count: usize,
    config: &ReconstructionConfig,
    cpu: Option<CpuCapabilities>,
    low: &mut [i32; 16],
    high: &mut [i32; 256],
) -> Result<(), ReconstructionError> {
    dequantize_into(macroblock, block_count, cpu, low, high)?;
    inverse_first_level(low, config)
}

fn reconstruct_low_into(
    macroblock: &QuantizedMacroblock,
    block_count: usize,
    config: &ReconstructionConfig,
    low: &mut [i32; 16],
) -> Result<(), ReconstructionError> {
    *low = dequantize_low(macroblock, block_count)?;
    inverse_first_level(low, config)
}

fn finish_component(
    dc_low_pass: &[[i32; 16]],
    high_pass: &mut [[i32; 256]],
    geometry: ReconstructionGeometry,
    config: &ReconstructionConfig,
    samples: &mut Vec<i32>,
) -> Result<(PlanarSamples, bool), ReconstructionError> {
    let ReconstructionGeometry {
        macroblocks_x,
        macroblocks_y,
        block_columns,
        block_rows,
    } = geometry;
    let width = macroblocks_x.checked_mul(block_columns * 4).ok_or(
        ReconstructionError::ArithmeticOverflow("reconstructed width"),
    )?;
    let height = macroblocks_y.checked_mul(block_rows * 4).ok_or(
        ReconstructionError::ArithmeticOverflow("reconstructed height"),
    )?;
    let output_length =
        width
            .checked_mul(height)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "reconstructed plane size",
            ))?;
    let output_reused = samples.capacity() >= output_length;
    second_level_transform_into(
        dc_low_pass,
        high_pass,
        macroblocks_x,
        macroblocks_y,
        block_columns,
        block_rows,
        samples,
    )?;
    if config.overlap != OverlapMode::None {
        apply_configured_overlap(
            samples,
            width,
            height,
            config,
            block_columns * 4,
            block_rows * 4,
        )?;
    }
    Ok((
        PlanarSamples {
            origin_x: config
                .macroblock_origin_x
                .checked_mul(u32::from(config.block_columns) * 4)
                .ok_or(ReconstructionError::ArithmeticOverflow("output x origin"))?,
            origin_y: config
                .macroblock_origin_y
                .checked_mul(u32::from(config.block_rows) * 4)
                .ok_or(ReconstructionError::ArithmeticOverflow("output y origin"))?,
            width: u32::try_from(width)
                .map_err(|_| ReconstructionError::ArithmeticOverflow("output width conversion"))?,
            height: u32::try_from(height)
                .map_err(|_| ReconstructionError::ArithmeticOverflow("output height conversion"))?,
            samples: core::mem::take(samples),
        },
        output_reused,
    ))
}

fn dequantize_into(
    macroblock: &QuantizedMacroblock,
    block_count: usize,
    cpu: Option<CpuCapabilities>,
    low: &mut [i32; 16],
    high: &mut [i32; 256],
) -> Result<(), ReconstructionError> {
    let QuantizerSet { high_pass, .. } = macroblock.quantizers;
    *low = dequantize_low(macroblock, block_count)?;
    let high_pass = quantizer(high_pass, "high-pass")?;
    high.fill(0);
    if macroblock.bands.has_high_pass() {
        let coefficient_count = block_count * 16;
        if let Some(cpu) = cpu {
            scale_coefficients(
                cpu,
                high_pass,
                &macroblock.high_pass[..coefficient_count],
                &mut high[..coefficient_count],
            )?;
        } else {
            for block in 0..block_count {
                for coefficient in 1..16 {
                    let index = block * 16 + coefficient;
                    high[index] = scale(high_pass, macroblock.high_pass[index])?;
                }
            }
        }
    }
    Ok(())
}

fn dequantize_low(
    macroblock: &QuantizedMacroblock,
    block_count: usize,
) -> Result<[i32; 16], ReconstructionError> {
    let QuantizerSet { dc, low_pass, .. } = macroblock.quantizers;
    let dc = quantizer(dc, "DC")?;
    let low_pass = quantizer(low_pass, "low-pass")?;
    let mut low = [0; 16];
    low[0] = scale(dc, macroblock.dc_low_pass[0])?;
    if macroblock.bands.has_low_pass() {
        for (output, &coefficient) in low[1..block_count]
            .iter_mut()
            .zip(&macroblock.dc_low_pass[1..block_count])
        {
            *output = scale(low_pass, coefficient)?;
        }
    }
    Ok(low)
}

fn component_block_geometry(
    config: &ReconstructionConfig,
) -> Result<(usize, usize), ReconstructionError> {
    let columns = usize::from(config.block_columns);
    let rows = usize::from(config.block_rows);
    match (columns, rows) {
        (4 | 2, 4) | (2, 2) => Ok((columns, rows)),
        _ => Err(ReconstructionError::InvalidPlaneGeometry(
            "unsupported component macroblock shape",
        )),
    }
}

fn inverse_first_level(
    low: &mut [i32; 16],
    config: &ReconstructionConfig,
) -> Result<(), ReconstructionError> {
    match (config.block_columns, config.block_rows) {
        (4, 4) => inverse_core_transform(low)?,
        (2, 4) => {
            let mut coefficients = [0; 8];
            coefficients.copy_from_slice(&low[..8]);
            inverse_chroma_422(&mut coefficients)?;
            low[..8].copy_from_slice(&coefficients);
        }
        (2, 2) => {
            let mut coefficients = [0; 4];
            coefficients.copy_from_slice(&low[..4]);
            inverse_chroma_420(&mut coefficients)?;
            low[..4].copy_from_slice(&coefficients);
        }
        _ => {
            return Err(ReconstructionError::InvalidPlaneGeometry(
                "unsupported first-level transform shape",
            ));
        }
    }
    if config.scale_after_first_transform {
        let count = usize::from(config.block_columns) * usize::from(config.block_rows);
        for coefficient in &mut low[..count] {
            *coefficient =
                coefficient
                    .checked_mul(2)
                    .ok_or(ReconstructionError::ArithmeticOverflow(
                        "scaled chroma normalization",
                    ))?;
        }
    }
    Ok(())
}

fn quantizer(step: u32, band: &'static str) -> Result<Quantizer, ReconstructionError> {
    Quantizer::new(step).ok_or(ReconstructionError::ZeroQuantizer(band))
}

fn scale(quantizer: Quantizer, coefficient: i32) -> Result<i32, ReconstructionError> {
    quantizer
        .dequantize(coefficient)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("coefficient dequantization"))
}

fn combine_low_pass_into(
    macroblocks: &[[i32; 16]],
    macroblocks_x: usize,
    macroblocks_y: usize,
    block_columns: usize,
    block_rows: usize,
    plane: &mut Vec<i32>,
) -> Result<(), ReconstructionError> {
    let width = macroblocks_x
        .checked_mul(block_columns)
        .ok_or(ReconstructionError::ArithmeticOverflow("low-pass width"))?;
    let length = width
        .checked_mul(macroblocks_y)
        .and_then(|value| value.checked_mul(block_rows))
        .ok_or(ReconstructionError::ArithmeticOverflow(
            "low-pass plane size",
        ))?;
    plane.clear();
    plane.resize(length, 0);
    for macroblock_y in 0..macroblocks_y {
        for macroblock_x in 0..macroblocks_x {
            let block = &macroblocks[macroblock_y * macroblocks_x + macroblock_x];
            for row in 0..block_rows {
                let destination =
                    (macroblock_y * block_rows + row) * width + macroblock_x * block_columns;
                plane[destination..destination + block_columns].copy_from_slice(
                    &block[row * block_columns..row * block_columns + block_columns],
                );
            }
        }
    }
    Ok(())
}

fn scatter_low_pass(
    plane: &[i32],
    macroblocks: &mut [[i32; 16]],
    macroblocks_x: usize,
    macroblocks_y: usize,
    block_columns: usize,
    block_rows: usize,
) {
    let width = macroblocks_x * block_columns;
    for macroblock_y in 0..macroblocks_y {
        for macroblock_x in 0..macroblocks_x {
            let block = &mut macroblocks[macroblock_y * macroblocks_x + macroblock_x];
            for row in 0..block_rows {
                let source =
                    (macroblock_y * block_rows + row) * width + macroblock_x * block_columns;
                block[row * block_columns..row * block_columns + block_columns]
                    .copy_from_slice(&plane[source..source + block_columns]);
            }
        }
    }
}

fn second_level_transform_into(
    low: &[[i32; 16]],
    high: &mut [[i32; 256]],
    macroblocks_x: usize,
    macroblocks_y: usize,
    block_columns: usize,
    block_rows: usize,
    plane: &mut Vec<i32>,
) -> Result<(), ReconstructionError> {
    let macroblock_width = block_columns * 4;
    let macroblock_height = block_rows * 4;
    let width = macroblocks_x * macroblock_width;
    let height = macroblocks_y * macroblock_height;
    if high.len() >= 64 {
        high.par_iter_mut()
            .zip(low.par_iter())
            .try_for_each(|(high, low)| {
                transform_macroblock(high, low, block_columns * block_rows)
            })?;
    } else {
        for (high, low) in high.iter_mut().zip(low) {
            transform_macroblock(high, low, block_columns * block_rows)?;
        }
    }
    let length = width
        .checked_mul(height)
        .ok_or(ReconstructionError::ArithmeticOverflow(
            "reconstructed plane size",
        ))?;
    plane.clear();
    plane.resize(length, 0);
    for macroblock_y in 0..macroblocks_y {
        for macroblock_x in 0..macroblocks_x {
            let index = macroblock_y * macroblocks_x + macroblock_x;
            for block in 0..block_columns * block_rows {
                let coefficients = &high[index][block * 16..block * 16 + 16];
                let block_x = macroblock_x * macroblock_width + block % block_columns * 4;
                let block_y = macroblock_y * macroblock_height + block / block_columns * 4;
                for row in 0..4 {
                    let destination = (block_y + row) * width + block_x;
                    plane[destination..destination + 4]
                        .copy_from_slice(&coefficients[row * 4..row * 4 + 4]);
                }
            }
        }
    }
    Ok(())
}

fn transform_macroblock(
    high: &mut [i32; 256],
    low: &[i32; 16],
    block_count: usize,
) -> Result<(), ReconstructionError> {
    for (coefficients, &dc) in high.chunks_exact_mut(16).zip(low).take(block_count) {
        let coefficients: &mut [i32; 16] = coefficients
            .try_into()
            .expect("a transform block contains sixteen coefficients");
        coefficients[0] = dc;
        inverse_core_transform(coefficients)?;
    }
    Ok(())
}

fn apply_configured_overlap(
    samples: &mut [i32],
    width: usize,
    height: usize,
    config: &ReconstructionConfig,
    horizontal_scale: usize,
    vertical_scale: usize,
) -> Result<(), ReconstructionError> {
    let columns = scale_boundaries(&config.tiles.column_boundaries, horizontal_scale)?;
    let rows = scale_boundaries(&config.tiles.row_boundaries, vertical_scale)?;
    apply_overlap(
        samples,
        width,
        height,
        &columns,
        &rows,
        config.tiles.hard_boundaries,
    )
}

fn apply_configured_subsampled_overlap(
    samples: &mut [i32],
    width: usize,
    height: usize,
    config: &ReconstructionConfig,
    horizontal_scale: usize,
    vertical_scale: usize,
) -> Result<(), ReconstructionError> {
    let columns = scale_boundaries(&config.tiles.column_boundaries, horizontal_scale)?;
    let rows = scale_boundaries(&config.tiles.row_boundaries, vertical_scale)?;
    apply_subsampled_first_overlap(
        samples,
        width,
        height,
        &columns,
        &rows,
        config.tiles.hard_boundaries,
    )
}

fn scale_boundaries(boundaries: &[u32], scale: usize) -> Result<Vec<usize>, ReconstructionError> {
    boundaries
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .ok()
                .and_then(|value| value.checked_mul(scale))
                .ok_or(ReconstructionError::ArithmeticOverflow(
                    "tile boundary scaling",
                ))
        })
        .collect()
}

fn validate_partition(
    boundaries: &[u32],
    extent: u32,
    axis: &'static str,
) -> Result<(), ReconstructionError> {
    if boundaries.len() < 2
        || boundaries.first() != Some(&0)
        || boundaries.last() != Some(&extent)
        || boundaries.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(ReconstructionError::InvalidTilePartition(axis));
    }
    Ok(())
}
