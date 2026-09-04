//! Typed output-storage alpha premultiplication.

use jxr_core::{DecodedSamples, DecodedSamplesMut, PixelFormat};
use jxr_math::alpha::{
    premultiply_f32, premultiply_sign_magnitude_15, premultiply_signed, premultiply_unsigned,
};

use super::OutputFormatError;

pub(super) fn premultiply_output(
    samples: &mut DecodedSamples,
    format: PixelFormat,
) -> Result<(), OutputFormatError> {
    let channels = usize::from(format.channel_count());
    match samples {
        DecodedSamples::U8(values) => premultiply_u8(values, channels)?,
        DecodedSamples::U16(values) => premultiply_u16(values, channels)?,
        DecodedSamples::I16(values) => premultiply_i16(values, channels)?,
        DecodedSamples::I32(values) => premultiply_i32(values, channels)?,
        DecodedSamples::F16(values) => premultiply_f16(values, channels)?,
        DecodedSamples::F32(values) => premultiply_float(values, channels),
        DecodedSamples::BitPacked(_)
        | DecodedSamples::Rgb555(_)
        | DecodedSamples::Rgb565(_)
        | DecodedSamples::Rgb101010(_)
        | DecodedSamples::Rgbe(_) => {
            return Err(OutputFormatError::UnsupportedCombination {
                combination: "alpha premultiplication for packed output",
            });
        }
    }
    Ok(())
}

pub(super) fn premultiply_output_mut(
    samples: DecodedSamplesMut<'_>,
    format: PixelFormat,
) -> Result<(), OutputFormatError> {
    let channels = usize::from(format.channel_count());
    match samples {
        DecodedSamplesMut::U8(values) => premultiply_u8(values, channels)?,
        DecodedSamplesMut::U16(values) => premultiply_u16(values, channels)?,
        DecodedSamplesMut::I16(values) => premultiply_i16(values, channels)?,
        DecodedSamplesMut::I32(values) => premultiply_i32(values, channels)?,
        DecodedSamplesMut::F16(values) => premultiply_f16(values, channels)?,
        DecodedSamplesMut::F32(values) => premultiply_float(values, channels),
        DecodedSamplesMut::BitPacked(_)
        | DecodedSamplesMut::Rgb555(_)
        | DecodedSamplesMut::Rgb565(_)
        | DecodedSamplesMut::Rgb101010(_)
        | DecodedSamplesMut::Rgbe(_) => {
            return Err(OutputFormatError::UnsupportedCombination {
                combination: "alpha premultiplication for packed output",
            });
        }
    }
    Ok(())
}

pub(super) fn premultiply_u8(values: &mut [u8], channels: usize) -> Result<(), OutputFormatError> {
    for pixel in values.chunks_exact_mut(channels) {
        let alpha = u32::from(pixel[channels - 1]);
        for value in &mut pixel[..channels - 1] {
            *value = u8::try_from(
                premultiply_unsigned(u32::from(*value), alpha, 255).map_err(math_error)?,
            )
            .expect("premultiplied u8 remains in range");
        }
    }
    Ok(())
}

fn premultiply_u16(values: &mut [u16], channels: usize) -> Result<(), OutputFormatError> {
    for pixel in values.chunks_exact_mut(channels) {
        let alpha = u32::from(pixel[channels - 1]);
        for value in &mut pixel[..channels - 1] {
            *value = u16::try_from(
                premultiply_unsigned(u32::from(*value), alpha, 65_535).map_err(math_error)?,
            )
            .expect("premultiplied u16 remains in range");
        }
    }
    Ok(())
}

fn premultiply_i16(values: &mut [i16], channels: usize) -> Result<(), OutputFormatError> {
    for pixel in values.chunks_exact_mut(channels) {
        let alpha = i32::from(pixel[channels - 1]);
        for value in &mut pixel[..channels - 1] {
            *value = i16::try_from(
                premultiply_signed(i32::from(*value), alpha, 32_767).map_err(math_error)?,
            )
            .expect("premultiplied i16 remains in range");
        }
    }
    Ok(())
}

fn premultiply_i32(values: &mut [i32], channels: usize) -> Result<(), OutputFormatError> {
    for pixel in values.chunks_exact_mut(channels) {
        let alpha = pixel[channels - 1];
        for value in &mut pixel[..channels - 1] {
            *value = premultiply_signed(*value, alpha, i32::MAX).map_err(math_error)?;
        }
    }
    Ok(())
}

fn premultiply_f16(values: &mut [u16], channels: usize) -> Result<(), OutputFormatError> {
    for pixel in values.chunks_exact_mut(channels) {
        let alpha = pixel[channels - 1];
        for value in &mut pixel[..channels - 1] {
            *value = premultiply_sign_magnitude_15(*value, alpha).map_err(math_error)?;
        }
    }
    Ok(())
}

fn premultiply_float(values: &mut [f32], channels: usize) {
    for pixel in values.chunks_exact_mut(channels) {
        let alpha = pixel[channels - 1];
        for value in &mut pixel[..channels - 1] {
            *value = premultiply_f32(*value, alpha);
        }
    }
}

fn math_error(_: jxr_math::MathError) -> OutputFormatError {
    OutputFormatError::arithmetic("premultiplying output alpha")
}
