use jxr_core::{DecodedSamplesMut, JxrError, JxrErrorKind, PixelFormat};

use crate::batch::{
    BatchDecodeOptions, BatchInfrastructureError, CpuBatchDestination, CpuBatchSamples,
    PreparedBatch, PreparedBatchGroup, prepare::try_vec,
};

pub(super) fn validate_prepared_count(
    prepared: &PreparedBatch,
    options: BatchDecodeOptions,
) -> Result<(), BatchInfrastructureError> {
    let requested = prepared.input_count();
    if requested > options.max_inputs {
        return Err(BatchInfrastructureError::TooManyInputs {
            requested,
            maximum: options.max_inputs,
        });
    }
    Ok(())
}

pub(super) fn validate_output_budget(
    prepared: &PreparedBatch,
    options: BatchDecodeOptions,
) -> Result<(), BatchInfrastructureError> {
    let requested = prepared.groups().iter().try_fold(0_u64, |total, group| {
        let stride = u64::try_from(group.info().image_stride_bytes()).ok()?;
        let images = u64::try_from(group.images().len()).ok()?;
        total.checked_add(stride.checked_mul(images)?)
    });
    let maximum = options
        .max_host_allocation_bytes
        .min(prepared.options().max_host_allocation_bytes);
    let Some(requested) = requested else {
        return Err(BatchInfrastructureError::OutputAllocationTooLarge {
            requested: u64::MAX,
            maximum,
        });
    };
    if requested > maximum {
        return Err(BatchInfrastructureError::OutputAllocationTooLarge { requested, maximum });
    }
    Ok(())
}

pub(super) fn validate_group_destination(
    group: &PreparedBatchGroup,
    destination: &CpuBatchDestination<'_>,
) -> Result<(), BatchInfrastructureError> {
    if !destination.matches_format(group.info().format()) {
        return Err(BatchInfrastructureError::InvalidDestination {
            reason: "destination variant differs from the prepared output format",
        });
    }
    let expected = group
        .info()
        .image_stride_elements()
        .checked_mul(group.images().len())
        .ok_or(BatchInfrastructureError::InvalidDestination {
            reason: "destination element count overflows usize",
        })?;
    if destination.len() != expected {
        return Err(BatchInfrastructureError::InvalidDestination {
            reason: "destination length differs from the prepared group extent",
        });
    }
    Ok(())
}

#[derive(Debug, Default)]
pub(super) struct CpuLayoutWorkspace {
    bytes: Vec<u8>,
    unsigned_words: Vec<u16>,
    signed_words: Vec<i16>,
    signed_dwords: Vec<i32>,
    floats: Vec<f32>,
    reuses: u64,
}

impl CpuLayoutWorkspace {
    pub(super) const fn reuses(&self) -> u64 {
        self.reuses
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.bytes
            .capacity()
            .saturating_add(self.unsigned_words.capacity().saturating_mul(2))
            .saturating_add(self.signed_words.capacity().saturating_mul(2))
            .saturating_add(self.signed_dwords.capacity().saturating_mul(4))
            .saturating_add(self.floats.capacity().saturating_mul(4))
    }

    pub(super) fn reorder_nchw(
        &mut self,
        destination: DecodedSamplesMut<'_>,
        dimensions: (u32, u32),
        channels: usize,
    ) -> Result<(), JxrError> {
        let pixels = usize::try_from(dimensions.0)
            .ok()
            .and_then(|width| {
                usize::try_from(dimensions.1)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| JxrError::arithmetic("NCHW pixel count"))?;
        let reused = match destination {
            DecodedSamplesMut::U8(values) => {
                reorder_values(values, pixels, channels, &mut self.bytes)?
            }
            DecodedSamplesMut::U16(values) | DecodedSamplesMut::F16(values) => {
                reorder_values(values, pixels, channels, &mut self.unsigned_words)?
            }
            DecodedSamplesMut::I16(values) => {
                reorder_values(values, pixels, channels, &mut self.signed_words)?
            }
            DecodedSamplesMut::I32(values) => {
                reorder_values(values, pixels, channels, &mut self.signed_dwords)?
            }
            DecodedSamplesMut::F32(values) => {
                reorder_values(values, pixels, channels, &mut self.floats)?
            }
            DecodedSamplesMut::BitPacked(_)
            | DecodedSamplesMut::Rgb555(_)
            | DecodedSamplesMut::Rgb565(_)
            | DecodedSamplesMut::Rgb101010(_)
            | DecodedSamplesMut::Rgbe(_) => {
                return Err(JxrError::new(
                    JxrErrorKind::InternalInvariant,
                    "NCHW destination storage",
                ));
            }
        };
        self.reuses = self.reuses.saturating_add(u64::from(reused));
        Ok(())
    }
}

fn reorder_values<Value: Copy + Default>(
    values: &mut [Value],
    pixels: usize,
    channels: usize,
    scratch: &mut Vec<Value>,
) -> Result<bool, JxrError> {
    let expected = pixels
        .checked_mul(channels)
        .ok_or_else(|| JxrError::arithmetic("NCHW sample count"))?;
    if values.len() != expected {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "NCHW sample count",
        ));
    }
    if channels <= 1 {
        return Ok(false);
    }
    let reused = scratch.capacity() >= values.len();
    scratch.resize(values.len(), Value::default());
    for pixel in 0..pixels {
        for channel in 0..channels {
            scratch[channel * pixels + pixel] = values[pixel * channels + channel];
        }
    }
    values.copy_from_slice(scratch);
    Ok(reused)
}

impl CpuBatchSamples {
    pub(super) fn zeroed(
        format: PixelFormat,
        elements: usize,
        what: &'static str,
    ) -> Result<Self, BatchInfrastructureError> {
        macro_rules! zeroed {
            ($type:ty, $variant:ident) => {{
                let mut values = try_vec::<$type>(elements, what)?;
                values.resize(elements, <$type>::default());
                Self::$variant(values)
            }};
        }
        Ok(match format {
            PixelFormat::BitPacked(_) => zeroed!(u8, BitPacked),
            PixelFormat::U8(_) => zeroed!(u8, U8),
            PixelFormat::U16(_) => zeroed!(u16, U16),
            PixelFormat::I16(_) => zeroed!(i16, I16),
            PixelFormat::I32(_) => zeroed!(i32, I32),
            PixelFormat::F16(_) => zeroed!(u16, F16),
            PixelFormat::F32(_) => zeroed!(f32, F32),
            PixelFormat::Rgb555 => zeroed!(u16, Rgb555),
            PixelFormat::Rgb565 => zeroed!(u16, Rgb565),
            PixelFormat::Rgb101010 => zeroed!(u32, Rgb101010),
            PixelFormat::Rgbe => zeroed!(u32, Rgbe),
        })
    }

    pub(super) fn destination_mut(&mut self) -> jxr_core::DecodedSamplesMut<'_> {
        match self {
            Self::BitPacked(values) => jxr_core::DecodedSamplesMut::BitPacked(values),
            Self::U8(values) => jxr_core::DecodedSamplesMut::U8(values),
            Self::U16(values) => jxr_core::DecodedSamplesMut::U16(values),
            Self::I16(values) => jxr_core::DecodedSamplesMut::I16(values),
            Self::I32(values) => jxr_core::DecodedSamplesMut::I32(values),
            Self::F16(values) => jxr_core::DecodedSamplesMut::F16(values),
            Self::F32(values) => jxr_core::DecodedSamplesMut::F32(values),
            Self::Rgb555(values) => jxr_core::DecodedSamplesMut::Rgb555(values),
            Self::Rgb565(values) => jxr_core::DecodedSamplesMut::Rgb565(values),
            Self::Rgb101010(values) => jxr_core::DecodedSamplesMut::Rgb101010(values),
            Self::Rgbe(values) => jxr_core::DecodedSamplesMut::Rgbe(values),
        }
    }

    pub(super) fn copy_within(&mut self, source: core::ops::Range<usize>, destination: usize) {
        match self {
            Self::BitPacked(values) | Self::U8(values) => values.copy_within(source, destination),
            Self::U16(values) | Self::F16(values) | Self::Rgb555(values) | Self::Rgb565(values) => {
                values.copy_within(source, destination);
            }
            Self::I16(values) => values.copy_within(source, destination),
            Self::I32(values) => values.copy_within(source, destination),
            Self::F32(values) => values.copy_within(source, destination),
            Self::Rgb101010(values) | Self::Rgbe(values) => {
                values.copy_within(source, destination);
            }
        }
    }

    pub(super) fn truncate(&mut self, length: usize) {
        match self {
            Self::BitPacked(values) | Self::U8(values) => values.truncate(length),
            Self::U16(values) | Self::F16(values) | Self::Rgb555(values) | Self::Rgb565(values) => {
                values.truncate(length);
            }
            Self::I16(values) => values.truncate(length),
            Self::I32(values) => values.truncate(length),
            Self::F32(values) => values.truncate(length),
            Self::Rgb101010(values) | Self::Rgbe(values) => values.truncate(length),
        }
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::{ChannelLayout, DecodedSamplesMut, PixelFormat};

    use super::{CpuBatchSamples, CpuLayoutWorkspace};

    #[test]
    fn native_batch_owner_preserves_every_decoded_storage_variant() {
        let cases = [
            PixelFormat::BitPacked(ChannelLayout::Luma),
            PixelFormat::U8(ChannelLayout::Luma),
            PixelFormat::U16(ChannelLayout::Luma),
            PixelFormat::I16(ChannelLayout::Luma),
            PixelFormat::I32(ChannelLayout::Luma),
            PixelFormat::F16(ChannelLayout::Luma),
            PixelFormat::F32(ChannelLayout::Luma),
            PixelFormat::Rgb555,
            PixelFormat::Rgb565,
            PixelFormat::Rgb101010,
            PixelFormat::Rgbe,
        ];
        for format in cases {
            let mut batch = CpuBatchSamples::zeroed(format, 1, "typed batch test").unwrap();
            let destination = batch.destination_mut();
            assert!(destination.matches_format(format), "{format:?}");
            assert_eq!(destination.len(), 1, "{format:?}");
        }
    }

    #[test]
    fn nchw_reorders_typed_channels_and_reuses_scratch() {
        let mut workspace = CpuLayoutWorkspace::default();
        let mut first = [1_u16, 2, 3, 4, 5, 6];
        workspace
            .reorder_nchw(DecodedSamplesMut::U16(&mut first), (2, 1), 3)
            .unwrap();
        assert_eq!(first, [1, 4, 2, 5, 3, 6]);
        assert_eq!(workspace.reuses(), 0);
        assert!(workspace.retained_bytes() >= 12);

        let mut second = [7_u16, 8, 9, 10, 11, 12];
        workspace
            .reorder_nchw(DecodedSamplesMut::F16(&mut second), (2, 1), 3)
            .unwrap();
        assert_eq!(second, [7, 10, 8, 11, 9, 12]);
        assert_eq!(workspace.reuses(), 1);
    }

    #[test]
    fn typed_batch_owner_exposes_and_compacts_its_final_allocation() {
        let mut samples =
            CpuBatchSamples::zeroed(PixelFormat::U16(ChannelLayout::Luma), 4, "typed batch test")
                .unwrap();
        let jxr_core::DecodedSamplesMut::U16(values) = samples.destination_mut() else {
            panic!("expected U16 destination")
        };
        values.copy_from_slice(&[1, 2, 3, 4]);
        samples.copy_within(2..4, 0);
        samples.truncate(2);
        assert_eq!(samples, CpuBatchSamples::U16(vec![3, 4]));
    }
}
