//! Source pixel formats and their conversion to signed component planes.
//!
//! This is the decoder's output stage run backwards. Nothing about a format
//! is defined here: the container format table says how a pixel format is
//! declared, [`ChannelScale`] holds each depth's bias and scaling, and the
//! float, RGBE, and colour conversions are the inverses defined beside the
//! decoder's own. A format therefore encodes exactly as it decodes.
//!
//! Rows convert independently of one another, so they are read in parallel:
//! each channel of a row through the scaling resolved once for the image,
//! and then the row's colour conversion in place.

use core::ops::Range;

use jxr_core::{
    AnnexABitDepth, AnnexAChannelOrder, AnnexAPixelFamily, ChromaSampling, ColorFormat,
    EncodeRequest, JxrError, JxrErrorKind, OutputBitDepth, PixelFormat, SourceImage, SourcePlane,
    SourceSamples,
};
use jxr_math::rgbe::{prescale_rgbe, unpack_rgbe};

use crate::{
    executor::Executor,
    output_format::{
        InterleavedLoad, InterleavedSample, OutputFormatError, bit_position,
        color::convert_forward,
        is_direct,
        scaling::{ChannelScale, float16_sample, float32_sample},
        unpack_rgb555, unpack_rgb565, unpack_rgb101010,
    },
    pixel_format::{FormatDeclaration, annex_a_format_for, output_color_code},
    tile_encode::MAX_COMPONENTS,
};

use super::layout::ImageLayout;
#[cfg(test)]
use crate::reconstruct::subsample_chroma;

/// One pixel's channels; entries beyond the format's component count are zero.
type Channels = [i32; MAX_COMPONENTS];

/// Most planes one image is read into: every colour component and alpha.
const MAX_PLANES: usize = MAX_COMPONENTS + 1;

/// Rows are grouped so that each parallel task reads at least this many
/// samples, and an image of fewer is read on the calling thread.
const SAMPLES_PER_TASK: usize = 64 * 1024;

/// Codestream and container syntax describing a supported source format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SourceFormat {
    /// Colour format a decoder reports and converts to.
    pub(super) output_color: ColorFormat,
    /// Colour format the components are coded in.
    pub(super) internal_color: ColorFormat,
    /// Colour format each source pixel converts to: the internal one, with
    /// its chroma still at the source's resolution.
    full_color: ColorFormat,
    pub(super) bit_depth: OutputBitDepth,
    pub(super) components: u16,
    /// `OUTPUT_CLR_FMT` code of `output_color`.
    pub(super) output_color_code: u8,
    /// Annex-A pixel-format GUID declaring the source format in a container.
    pub(super) annex_a_guid: [u8; 16],
    /// Whether each pixel ends with an alpha sample, which is coded as a
    /// Y-only plane of its own rather than as a component.
    pub(super) alpha: bool,
    /// Whether the colour samples are declared premultiplied by alpha.
    pub(super) premultiplied_alpha: bool,
    /// Whether an interleaved pixel stores its three colour channels in
    /// reverse of the output colour's order, blue first.
    reversed: bool,
    /// Storage elements per interleaved pixel, including any padding channel.
    pixel_samples: usize,
}

impl SourceFormat {
    /// Describe `format` as the container format table declares it, with
    /// its colour samples premultiplied by alpha or not and its samples
    /// `sample_bits` wide, rejecting formats without an implemented forward
    /// path.
    pub(super) fn resolve(
        format: PixelFormat,
        premultiplied_alpha: bool,
        sample_bits: Option<u8>,
        request: &EncodeRequest,
    ) -> Result<Self, JxrError> {
        let declaration = FormatDeclaration {
            cmyk_direct: request.cmyk_direct,
            premultiplied_alpha,
            sample_bits,
        };
        let (annex_a_guid, descriptor) =
            annex_a_format_for(format, declaration).ok_or_else(unsupported)?;
        let full_chroma = ColorFormat::Yuv(ChromaSampling::Cs444);
        // `(output, per-pixel, internal)`: full-resolution chroma is coded at
        // the sampling the request selects.
        let (output_color, full_color, internal_color) = match descriptor.family {
            AnnexAPixelFamily::Luma => (ColorFormat::Luma, ColorFormat::Luma, ColorFormat::Luma),
            AnnexAPixelFamily::Rgb => (
                ColorFormat::Rgb,
                full_chroma,
                ColorFormat::Yuv(request.chroma_sampling),
            ),
            AnnexAPixelFamily::Rgbe => (
                ColorFormat::Rgbe,
                full_chroma,
                ColorFormat::Yuv(request.chroma_sampling),
            ),
            AnnexAPixelFamily::Cmyk { direct } => {
                let output = if direct {
                    ColorFormat::CmykDirect
                } else {
                    ColorFormat::Cmyk
                };
                (output, ColorFormat::YuvK, ColorFormat::YuvK)
            }
            // The channels are the coded components themselves, apart from
            // an alpha channel.
            AnnexAPixelFamily::NComponent => {
                let channels = descriptor.channels - u8::from(descriptor.alpha);
                let color = ColorFormat::NComponent(u16::from(channels));
                (color, color, color)
            }
            AnnexAPixelFamily::Yuv(ChromaSampling::Cs444) => (
                full_chroma,
                full_chroma,
                ColorFormat::Yuv(request.chroma_sampling),
            ),
            // Chroma the source has already subsampled is coded as it is.
            AnnexAPixelFamily::Yuv(sampling) => {
                let color = ColorFormat::Yuv(sampling);
                (color, color, color)
            }
        };
        let components = internal_color
            .component_count()
            .filter(|&count| usize::from(count) <= MAX_COMPONENTS)
            .ok_or_else(unsupported)?;
        let shift_bits = request.shift_bits.unwrap_or(match descriptor.bit_depth {
            AnnexABitDepth::I32 => EncodeRequest::DEFAULT_I32_SHIFT_BITS,
            _ => 0,
        });
        let bit_depth = OutputBitDepth::from_annex_a(
            descriptor.bit_depth,
            shift_bits,
            request.float_mantissa_bits,
            request.float_exponent_bias,
        );
        let reversed = matches!(
            descriptor.order,
            AnnexAChannelOrder::Bgr | AnnexAChannelOrder::Bgrx | AnnexAChannelOrder::Bgra
        );
        Ok(Self {
            output_color,
            internal_color,
            full_color,
            bit_depth,
            components,
            output_color_code: output_color_code(descriptor.family),
            annex_a_guid,
            alpha: descriptor.alpha,
            premultiplied_alpha: descriptor.premultiplied_alpha,
            reversed,
            // The storage layout, not the descriptor, counts a padding channel.
            pixel_samples: usize::from(format.channel_count()),
        })
    }

    /// Whether the samples of a pixel share one packed storage word, whose
    /// red and blue fields the codestream then carries in output order.
    pub(super) const fn is_packed(self) -> bool {
        matches!(
            self.bit_depth,
            OutputBitDepth::Rgb555 | OutputBitDepth::Rgb565 | OutputBitDepth::Rgb101010
        )
    }

    /// The reconstructed-sample scaling of output channel `channel`.
    fn scale(self, channel: usize, scaled: bool) -> ChannelScale {
        // RGBE mantissas are coded as eight-bit samples without a bias.
        ChannelScale::resolve(channel, self.output_color, self.bit_depth, scaled)
    }

    /// Offset of output channel `channel` within an interleaved pixel.
    const fn channel_offset(self, channel: usize) -> usize {
        if self.reversed && channel < 3 {
            2 - channel
        } else {
            channel
        }
    }

    /// The sampling the encoder reduces full-resolution chroma to, when the
    /// components are coded with coarser chroma than the source has.
    pub(super) fn decimation(self) -> Option<ChromaSampling> {
        match self.internal_color {
            ColorFormat::Yuv(sampling) if self.internal_color != self.full_color => Some(sampling),
            _ => None,
        }
    }

    /// Convert a row of pixels to the coded components in place, at the
    /// pixels' own resolution: `rows` holds that row of every colour plane,
    /// the output-colour channels going in.
    fn internal_rows(self, rows: &mut [&mut [i32]]) -> Result<(), OutputFormatError> {
        if is_direct(self.full_color, self.output_color) {
            return Ok(());
        }
        // Every conversion is between three or four channels; fixing the
        // count leaves the per-pixel work free of loops.
        match rows {
            [first, second, third] => self.convert_pixels([first, second, third]),
            [first, second, third, fourth] => self.convert_pixels([first, second, third, fourth]),
            _ => Err(OutputFormatError::UnsupportedCombination {
                combination: "output-to-internal colour conversion",
            }),
        }
    }

    /// [`convert_forward`] over a row of pixels of `N` channels, in place.
    fn convert_pixels<const N: usize>(
        self,
        rows: [&mut [i32]; N],
    ) -> Result<(), OutputFormatError> {
        let width = rows.iter().map(|row| row.len()).min().unwrap_or(0);
        let mut rows = rows.map(|row| &mut row[..width]);
        for x in 0..width {
            let mut channels = [0_i32; 4];
            for (channel, row) in channels.iter_mut().zip(&rows) {
                *channel = row[x];
            }
            let converted = convert_forward(self.full_color, self.output_color, channels)?;
            for (row, value) in rows.iter_mut().zip(converted) {
                row[x] = value;
            }
        }
        Ok(())
    }
}

fn unsupported() -> JxrError {
    JxrError::new(JxrErrorKind::Unsupported, "encode source pixel format")
}

/// The signed, macroblock-aligned planes of a source image, or rows of them.
#[derive(Debug, Default)]
pub(super) struct SourcePlanes {
    /// One plane per coded colour component.
    pub(super) colour: Vec<Vec<i32>>,
    /// The alpha channel of a format that has one.
    pub(super) alpha: Option<Vec<i32>>,
}

/// Which planes of a source image one reading builds.
///
/// The colour planes and the alpha plane are read apart, so the alpha plane
/// does not have to exist while the colour planes are transformed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PlaneSet {
    /// The planes of the coded colour components.
    Colour,
    /// The alpha plane of a format that has one.
    Alpha,
    /// Colour and alpha in one reading.
    All,
}

impl PlaneSet {
    /// Whether the reading builds the colour planes.
    pub(super) const fn colour(self) -> bool {
        !matches!(self, Self::Alpha)
    }

    /// Whether the reading builds the alpha plane of a format with one.
    pub(super) const fn alpha(self) -> bool {
        !matches!(self, Self::Colour)
    }
}

/// Whether each image plane is coded with scaled arithmetic, which decides
/// the reconstructed samples its source values come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PlaneScaling {
    pub(super) colour: bool,
    pub(super) alpha: bool,
}

/// Build the signed, macroblock-aligned planes `set` names of a source
/// image, each colour plane at the resolution its component is coded at:
/// [`SourceReader`] given the whole of every plane as one window, and
/// [`subsample_chroma`] given the whole of each chroma plane it subsamples.
/// The planes `set` leaves out are absent from the result.
#[cfg(test)]
pub(super) fn component_planes(
    image: &SourceImage<'_>,
    layout: &ImageLayout,
    format: SourceFormat,
    scaled: PlaneScaling,
    set: PlaneSet,
) -> Result<SourcePlanes, JxrError> {
    let mut planes = SourcePlanes::default();
    if !set.colour() && !format.alpha {
        return Ok(planes);
    }
    check_sample_depth(image)?;
    let reader = SourceReader::new(image, layout, format, scaled, set, Executor::DEFAULT);
    let windows = (0..=usize::from(format.components))
        .map(|plane| reader.plane_dimensions(plane).map(|(_, height)| 0..height))
        .collect::<Result<Vec<_>, _>>()?;
    reader.read(&windows, &vec![0; windows.len()], &mut planes)?;
    if let Some(sampling) = format.decimation().filter(|_| set.colour()) {
        let (width, _) = layout.plane_dimensions()?;
        Executor::DEFAULT.try_for_each(&mut planes.colour[1..], |_, plane| {
            *plane = subsample_chroma(plane, width, sampling)
                .map_err(|_| JxrError::arithmetic("encode chroma subsampling"))?;
            Ok::<(), JxrError>(())
        })?;
    }
    Ok(planes)
}

/// Reads the rows of a source image's planes that the encoder asks for, a
/// window of rows at a time.
pub(super) struct SourceReader<'a> {
    reader: PixelReader<'a>,
}

impl<'a> SourceReader<'a> {
    /// A reader of the planes `set` names of `image`, whose rows are read
    /// side by side through `executor`. The encoder checks the source's
    /// declared sample depth once before constructing its readers.
    pub(super) fn new(
        image: &'a SourceImage<'a>,
        layout: &'a ImageLayout,
        format: SourceFormat,
        scaled: PlaneScaling,
        set: PlaneSet,
        executor: Executor,
    ) -> Self {
        Self {
            reader: PixelReader {
                image,
                layout,
                format,
                set,
                executor,
                channel_scales: (0..usize::from(format.components))
                    .map(|channel| format.scale(channel, scaled.colour))
                    .collect(),
                alpha_scale: ChannelScale::alpha(format.bit_depth, scaled.alpha),
                scaled,
            },
        }
    }

    /// Padded width and height of plane `plane` as the source stores it:
    /// colour plane `plane`, or alpha for the index after the last colour
    /// plane. A planar source's subsampled planes are padded to the same
    /// fraction of the macroblock-aligned luma plane.
    pub(super) fn plane_dimensions(&self, plane: usize) -> Result<(usize, usize), JxrError> {
        self.reader.padded_dimensions(plane)
    }

    /// Colour planes of the source, which is also the index of its alpha
    /// plane.
    pub(super) fn colour_components(&self) -> usize {
        usize::from(self.reader.format.components)
    }

    /// Fill `planes` with the rows `windows[plane]` of each plane the
    /// reading builds, at the resolution the source stores it, indexed as
    /// [`Self::plane_dimensions`] indexes them. Each plane holds exactly
    /// its window's rows afterwards, in storage it already had where it
    /// can; a plane the reading leaves out is absent.
    ///
    /// The first `held[plane]` rows of a window are not read: the caller
    /// has already put them at the start of the plane, as rows read for an
    /// earlier window that overlapped this one, and they stay as they are.
    ///
    /// Rows below the image repeat its last row, and columns beyond it its
    /// last column. The planes of an interleaved or packed source are read
    /// together and take the same window and the same held rows.
    pub(super) fn read(
        &self,
        windows: &[Range<usize>],
        held: &[usize],
        planes: &mut SourcePlanes,
    ) -> Result<(), JxrError> {
        let reader = &self.reader;
        let format = reader.format;
        let stride = reader.image.row_stride;
        let windows = PlaneWindows { windows, held };
        match reader.image.samples {
            SourceSamples::U8(samples) => reader.stored_integers(samples, windows, planes),
            SourceSamples::U16(samples) => reader.stored_integers(samples, windows, planes),
            SourceSamples::I16(samples) => reader.integers(samples, windows, planes),
            SourceSamples::I32(samples) => reader.integers(samples, windows, planes),
            SourceSamples::F16(samples) => reader.samples(
                samples,
                |_, scaled, bits| float16_sample(bits, scaled),
                windows,
                planes,
            ),
            SourceSamples::F32(samples) => {
                let OutputBitDepth::F32 {
                    mantissa_length,
                    exponent_bias,
                } = format.bit_depth
                else {
                    return Err(mismatched_storage());
                };
                reader.samples(
                    samples,
                    |_, scaled, value| {
                        float32_sample(value.to_bits(), scaled, mantissa_length, exponent_bias)
                    },
                    windows,
                    planes,
                )
            }
            SourceSamples::BitPacked(samples) => reader.packed(
                |x, y| {
                    let (byte, shift) = bit_position(x);
                    [i32::from(samples[y * stride + byte] >> shift & 1), 0, 0]
                },
                windows,
                planes,
            ),
            SourceSamples::Rgb555(samples) => reader.packed(
                |x, y| unpack_rgb555(samples[y * stride + x]),
                windows,
                planes,
            ),
            SourceSamples::Rgb565(samples) => reader.packed(
                |x, y| unpack_rgb565(samples[y * stride + x]),
                windows,
                planes,
            ),
            SourceSamples::Rgb101010(samples) => reader.packed(
                |x, y| unpack_rgb101010(samples[y * stride + x]),
                windows,
                planes,
            ),
            SourceSamples::Rgbe(samples) => reader.packed(
                |x, y| prescale_rgbe(unpack_rgbe(samples[y * stride + x])),
                windows,
                planes,
            ),
            _ => Err(unsupported()),
        }
    }
}

/// The window of rows of each plane, and how many of its first rows the
/// plane already holds, indexed as [`SourceReader::plane_dimensions`]
/// indexes the planes.
#[derive(Clone, Copy)]
struct PlaneWindows<'a> {
    windows: &'a [Range<usize>],
    held: &'a [usize],
}

impl PlaneWindows<'_> {
    fn of(self, plane: usize) -> Result<(Range<usize>, usize), JxrError> {
        let window = self.windows.get(plane).cloned();
        let held = self.held.get(plane).copied();
        window
            .zip(held)
            .filter(|(window, held)| *held <= window.len())
            .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "encode source window"))
    }

    /// The one window of planes read together: those of every plane of
    /// `planes` read.
    fn shared(
        self,
        planes: impl IntoIterator<Item = usize>,
    ) -> Result<(Range<usize>, usize), JxrError> {
        let mut planes = planes.into_iter();
        let Some(first) = planes.next() else {
            return Ok((0..0, 0));
        };
        let window = self.of(first)?;
        for plane in planes {
            if self.of(plane)? != window {
                return Err(JxrError::new(
                    JxrErrorKind::InternalInvariant,
                    "encode windows of planes read together",
                ));
            }
        }
        Ok(window)
    }
}

/// Reads source pixels as the reconstructed output-colour channels a decoder
/// would scale back into them.
struct PixelReader<'a> {
    image: &'a SourceImage<'a>,
    layout: &'a ImageLayout,
    format: SourceFormat,
    /// The planes this reading builds.
    set: PlaneSet,
    /// Where the rows of a window are read side by side.
    executor: Executor,
    /// Scaling of each output channel, one entry per component.
    channel_scales: Vec<ChannelScale>,
    /// Scaling of the alpha channel, the one a decoder packs alpha with.
    alpha_scale: ChannelScale,
    scaled: PlaneScaling,
}

impl PixelReader<'_> {
    /// See [`SourceReader::plane_dimensions`].
    fn padded_dimensions(&self, plane: usize) -> Result<(usize, usize), JxrError> {
        let padded = self.layout.plane_dimensions()?;
        if self.image.planes.is_none() {
            return Ok(padded);
        }
        let (width, height) = self.image.plane_dimensions(plane);
        let halved = |padded: usize, full: u32, own: u32| {
            if own == full { padded } else { padded / 2 }
        };
        Ok((
            halved(padded.0, self.image.width, width),
            halved(padded.1, self.image.height, height),
        ))
    }

    /// Undo the output scaling of one pixel's integer channel values, of
    /// which a format with fewer than three channels uses the first.
    fn unscale(&self, values: [i32; 3]) -> Result<Channels, OutputFormatError> {
        let mut samples = [0_i32; MAX_COMPONENTS];
        for (sample, (scale, value)) in samples
            .iter_mut()
            .zip(self.channel_scales.iter().zip(values))
        {
            *sample = scale.unapply(value)?;
        }
        Ok(samples)
    }

    /// Planes of unsigned integer samples, through the decoder's interleaved
    /// store run backwards where it stores the format, and as
    /// [`Self::integers`] reads them otherwise.
    fn stored_integers<T: InterleavedSample>(
        &self,
        samples: &[T],
        windows: PlaneWindows<'_>,
        planes: &mut SourcePlanes,
    ) -> Result<(), JxrError> {
        match self.interleaved_load::<T>() {
            Some(load) => self.load_interleaved(samples, load, windows, planes),
            None => self.integers(samples, windows, planes),
        }
    }

    /// The decoder's interleaved load of this image's pixels stored as `T`,
    /// where it has one. Every colour channel of its layouts shares one
    /// scale.
    fn interleaved_load<T: InterleavedSample>(&self) -> Option<InterleavedLoad> {
        let format = self.format;
        if self.image.planes.is_some() {
            return None;
        }
        InterleavedLoad::resolve::<T>(
            (format.full_color, format.output_color),
            self.image.format,
            format.bit_depth,
            self.channel_scales.first()?,
            &self.alpha_scale,
        )
        .filter(|load| load.has_alpha() == format.alpha)
    }

    /// Planes of interleaved pixels read through `load`, which converts
    /// their colour as it reads them.
    fn load_interleaved<T: InterleavedSample>(
        &self,
        samples: &[T],
        load: InterleavedLoad,
        windows: PlaneWindows<'_>,
        planes: &mut SourcePlanes,
    ) -> Result<(), JxrError> {
        let (width, _) = self.dimensions()?;
        let row_samples = width * self.format.pixel_samples;
        let components = self.colour_planes();
        let read = |y: usize, rows: &mut [&mut [i32]]| {
            let stored = &samples[y * self.image.row_stride..][..row_samples];
            let (colour, alpha) = rows.split_at_mut(components);
            load.load_row(stored, colour, alpha.first_mut().map(|row| &mut **row));
            Ok(())
        };
        self.fill(self.format.alpha, read, false, windows, planes)
    }

    /// Planes of integer samples of any width, interleaved or planar.
    fn integers<T: Copy + Into<i32> + Sync>(
        &self,
        samples: &[T],
        windows: PlaneWindows<'_>,
        planes: &mut SourcePlanes,
    ) -> Result<(), JxrError> {
        self.samples(
            samples,
            |scale, _, value| scale.unapply(value.into()),
            windows,
            planes,
        )
    }

    /// Planes of an image that stores one sample per channel, interleaved or
    /// planar. `sample` converts each: it receives the scaling of the
    /// sample's channel, whether the channel's plane is scaled, and the
    /// stored value.
    fn samples<T: Copy + Sync>(
        &self,
        samples: &[T],
        sample: impl Fn(&ChannelScale, bool, T) -> Result<i32, OutputFormatError> + Sync,
        windows: PlaneWindows<'_>,
        planes: &mut SourcePlanes,
    ) -> Result<(), JxrError> {
        match &self.image.planes {
            Some(located) => self.planar(samples, located, sample, windows, planes),
            None => self.interleaved(samples, sample, windows, planes),
        }
    }

    /// Planes of a planar image, each read from its own plane of the source
    /// and padded to whole macroblocks of its own sampling. `sample` is as
    /// for [`Self::samples`].
    fn planar<T: Copy + Sync>(
        &self,
        samples: &[T],
        located: &[SourcePlane; 4],
        sample: impl Fn(&ChannelScale, bool, T) -> Result<i32, OutputFormatError> + Sync,
        windows: PlaneWindows<'_>,
        planes: &mut SourcePlanes,
    ) -> Result<(), JxrError> {
        let components = usize::from(self.format.components);
        let read = |plane: usize, output: &mut Vec<i32>, scale: &ChannelScale, scaled: bool| {
            let dimensions = self.image.plane_dimensions(plane);
            let width = usize::try_from(dimensions.0)
                .map_err(|_| JxrError::arithmetic("source plane width"))?;
            let height = usize::try_from(dimensions.1)
                .map_err(|_| JxrError::arithmetic("source plane height"))?;
            let (padded_width, _) = self.padded_dimensions(plane)?;
            let (window, held) = windows.of(plane)?;
            let SourcePlane { offset, row_stride } = located[plane];
            let mut output = [core::mem::take(output)];
            fill_rows(
                &mut output,
                (width, height),
                padded_width,
                (window, held),
                self.executor,
                |y, rows| {
                    let stored = &samples[offset + y * row_stride..][..width];
                    rows.iter_mut().try_for_each(|row| {
                        read_channel(stored, row, |value| sample(scale, scaled, value))
                    })
                },
            )?;
            let [filled] = output;
            Ok::<_, JxrError>(filled)
        };
        if self.set.colour() {
            planes.colour.resize_with(components, Vec::new);
            for (plane, output) in planes.colour.iter_mut().enumerate() {
                *output = read(
                    plane,
                    output,
                    &self.channel_scales[plane],
                    self.scaled.colour,
                )?;
            }
        } else {
            planes.colour.clear();
        }
        planes.alpha = if self.format.alpha && self.set.alpha() {
            let mut output = planes.alpha.take().unwrap_or_default();
            output = read(
                components,
                &mut output,
                &self.alpha_scale,
                self.scaled.alpha,
            )?;
            Some(output)
        } else {
            None
        };
        Ok(())
    }

    /// Planes of interleaved samples, read one channel of one row at a time.
    /// `sample` is as for [`Self::samples`].
    fn interleaved<T: Copy + Sync>(
        &self,
        samples: &[T],
        sample: impl Fn(&ChannelScale, bool, T) -> Result<i32, OutputFormatError> + Sync,
        windows: PlaneWindows<'_>,
        planes: &mut SourcePlanes,
    ) -> Result<(), JxrError> {
        let format = self.format;
        let (width, _) = self.dimensions()?;
        let row_samples = width * format.pixel_samples;
        // One channel of a row, from its offset within the pixel, into one
        // plane row.
        let channel = |stored: &[T], offset: usize, row: &mut [i32], scale, scaled| {
            let stored = stored[offset..].iter().step_by(format.pixel_samples);
            read_channel(stored, row, |value| sample(scale, scaled, value))
        };
        let read = |y: usize, rows: &mut [&mut [i32]]| {
            let stored = &samples[y * self.image.row_stride..][..row_samples];
            let (colour, alpha) = rows.split_at_mut(self.colour_planes());
            for (index, (row, scale)) in colour.iter_mut().zip(&self.channel_scales).enumerate() {
                let offset = format.channel_offset(index);
                channel(stored, offset, row, scale, self.scaled.colour)?;
            }
            // Alpha is the last sample of each pixel in every channel order.
            for row in alpha {
                let offset = format.pixel_samples - 1;
                channel(stored, offset, row, &self.alpha_scale, self.scaled.alpha)?;
            }
            Ok(())
        };
        self.fill(format.alpha, read, true, windows, planes)
    }

    /// Planes of pixels whose channels share storage elements, which
    /// `pixel` splits into the stored channel values of pixel `(x, y)`.
    fn packed(
        &self,
        pixel: impl Fn(usize, usize) -> [i32; 3] + Sync,
        windows: PlaneWindows<'_>,
        planes: &mut SourcePlanes,
    ) -> Result<(), JxrError> {
        let read = |y: usize, rows: &mut [&mut [i32]]| {
            let width = rows.first().map_or(0, |row| row.len());
            for x in 0..width {
                let samples = self.unscale(pixel(x, y))?;
                for (row, sample) in rows.iter_mut().zip(samples) {
                    row[x] = sample;
                }
            }
            Ok(())
        };
        self.fill(false, read, true, windows, planes)
    }

    /// Colour planes this reading builds: every component's, or none.
    fn colour_planes(&self) -> usize {
        if self.set.colour() {
            usize::from(self.format.components)
        } else {
            0
        }
    }

    /// Fill the planes this reading builds from `read`: every component
    /// plane, and an alpha plane when `alpha`, as far as the reading's
    /// [`PlaneSet`] has them, each with the one window they share. `read`
    /// receives an image row's number and that row of each plane, alpha
    /// last, to fill with the reconstructed output-colour channels of the
    /// row's pixels, which are then converted to the coded components when
    /// `convert`, or with the coded components themselves otherwise.
    fn fill(
        &self,
        alpha: bool,
        read: impl Fn(usize, &mut [&mut [i32]]) -> Result<(), OutputFormatError> + Sync,
        convert: bool,
        windows: PlaneWindows<'_>,
        planes: &mut SourcePlanes,
    ) -> Result<(), JxrError> {
        let format = self.format;
        let components = self.colour_planes();
        let alpha = alpha && self.set.alpha();
        let alpha_index = usize::from(format.components);
        let window = windows.shared((0..components).chain(alpha.then_some(alpha_index)))?;
        let (padded_width, _) = self.layout.plane_dimensions()?;
        let mut filled: Vec<Vec<i32>> = core::mem::take(&mut planes.colour);
        filled.truncate(components);
        filled.resize_with(components, Vec::new);
        if alpha {
            filled.push(planes.alpha.take().unwrap_or_default());
        } else {
            planes.alpha = None;
        }
        fill_rows(
            &mut filled,
            self.dimensions()?,
            padded_width,
            window,
            self.executor,
            |y, rows| {
                read(y, rows)?;
                if components == 0 || !convert {
                    return Ok(());
                }
                format.internal_rows(&mut rows[..components])
            },
        )?;
        if alpha {
            planes.alpha = filled.pop();
        }
        planes.colour = filled;
        Ok(())
    }

    /// Width and height of the image in samples.
    fn dimensions(&self) -> Result<(usize, usize), JxrError> {
        let width = usize::try_from(self.image.width)
            .map_err(|_| JxrError::arithmetic("source image width"))?;
        let height = usize::try_from(self.image.height)
            .map_err(|_| JxrError::arithmetic("source image height"))?;
        Ok((width, height))
    }
}

/// Convert the stored samples of one channel of a row into its plane row.
#[inline]
fn read_channel<'a, T: Copy + 'a>(
    stored: impl IntoIterator<Item = &'a T>,
    row: &mut [i32],
    sample: impl Fn(T) -> Result<i32, OutputFormatError>,
) -> Result<(), OutputFormatError> {
    for (output, &value) in row.iter_mut().zip(stored) {
        *output = sample(value)?;
    }
    Ok(())
}

/// One image row in every plane being filled: the samples the image covers,
/// and the samples beyond its width that pad the row to whole macroblocks.
#[derive(Default)]
struct PlaneRows<'a> {
    image: [&'a mut [i32]; MAX_PLANES],
    padding: [&'a mut [i32]; MAX_PLANES],
}

/// Fill the rows `window` of `planes`, whose rows are `padded_width`
/// samples long, from an image of `width` by `height` samples; each plane
/// is resized to hold exactly those rows, of which the first `held` are
/// left as they are.
///
/// `read` receives an image row's number and that row of each plane, `width`
/// samples long, to fill. Rows are read side by side through `executor`
/// unless the window is smaller than one task, and a failure is that of the
/// first row that has one. Samples beyond the image repeat its last column
/// and row: a row of the window below the image is read as the image's last.
fn fill_rows(
    planes: &mut [Vec<i32>],
    (width, height): (usize, usize),
    padded_width: usize,
    (window, held): (Range<usize>, usize),
    executor: Executor,
    read: impl Fn(usize, &mut [&mut [i32]]) -> Result<(), OutputFormatError> + Sync,
) -> Result<(), JxrError> {
    let count = planes.len();
    let length = padded_width
        .checked_mul(window.len())
        .ok_or_else(|| JxrError::arithmetic("encode plane size"))?;
    let unread = window.start + held..window.end;
    let mut rows: Vec<PlaneRows<'_>> = unread.clone().map(|_| PlaneRows::default()).collect();
    for (index, plane) in planes.iter_mut().enumerate() {
        // Every sample is written below, so storage kept from an earlier
        // window is not cleared first, and it grows to the window exactly.
        plane.reserve_exact(length.saturating_sub(plane.len()));
        plane.resize(length, 0);
        let samples = plane[held * padded_width..].chunks_exact_mut(padded_width);
        for (row, samples) in rows.iter_mut().zip(samples) {
            (row.image[index], row.padding[index]) = samples.split_at_mut(width);
        }
    }
    let last = height.saturating_sub(1);
    let fill = |(index, row): (usize, &mut PlaneRows<'_>)| {
        if let Err(error) = read((unread.start + index).min(last), &mut row.image[..count]) {
            return Some(error);
        }
        for (image, padding) in row.image.iter().zip(&mut row.padding) {
            if let Some(&last) = image.last() {
                padding.fill(last);
            }
        }
        None
    };
    let rows_per_task = (SAMPLES_PER_TASK / (width * count).max(1)).max(1);
    let failure = executor
        .if_worth(rows.len() > rows_per_task)
        .find_map_first(&mut rows, rows_per_task, |index, row| fill((index, row)));
    match failure {
        Some(error) => Err(source_error(&error)),
        None => Ok(()),
    }
}

/// Reject a sample beyond the depth its image declares, which a decoder
/// would clip instead of returning.
pub(super) fn check_sample_depth(image: &SourceImage<'_>) -> Result<(), JxrError> {
    let (Some(bits), SourceSamples::U16(samples)) = (image.sample_bits, image.samples) else {
        return Ok(());
    };
    let limit = 1_u32 << bits;
    let size = |value: u32| {
        usize::try_from(value).map_err(|_| JxrError::arithmetic("source image dimensions"))
    };
    // `(plane, samples in each of its rows, rows)` of every plane read.
    let mut planes = Vec::with_capacity(4);
    if let Some(located) = &image.planes {
        let count = image.plane_count().unwrap_or(0);
        for (index, &plane) in located.iter().enumerate().take(count) {
            let (width, height) = image.plane_dimensions(index);
            planes.push((plane, size(width)?, size(height)?));
        }
    } else {
        let plane = SourcePlane {
            offset: 0,
            row_stride: image.row_stride,
        };
        planes.push((plane, image.row_samples()?, size(image.height)?));
    }
    let within = planes.into_iter().all(|(plane, row_samples, rows)| {
        (0..rows).all(|row| {
            let start = plane.offset + row * plane.row_stride;
            samples[start..start + row_samples]
                .iter()
                .all(|&sample| u32::from(sample) < limit)
        })
    });
    if within {
        Ok(())
    } else {
        Err(JxrError::new(
            JxrErrorKind::InvalidRequest,
            "source sample exceeds its declared depth",
        ))
    }
}

fn mismatched_storage() -> JxrError {
    JxrError::new(
        JxrErrorKind::InvalidRequest,
        "source sample storage differs from its pixel format",
    )
}

fn source_error(error: &OutputFormatError) -> JxrError {
    let kind = match error {
        OutputFormatError::UnsupportedCombination { .. } => JxrErrorKind::Unsupported,
        OutputFormatError::InvalidFloatingPointSample => JxrErrorKind::InvalidRequest,
        OutputFormatError::ArithmeticOverflow { .. } => JxrErrorKind::ArithmeticOverflow,
        OutputFormatError::ComponentCount { .. }
        | OutputFormatError::InvalidPlane { .. }
        | OutputFormatError::CropOutsidePlane { .. }
        | OutputFormatError::AlphaMismatch => JxrErrorKind::InternalInvariant,
    };
    JxrError::new(kind, "encode source samples")
}

#[cfg(test)]
mod tests {
    use jxr_core::{ChannelLayout, DecodedSamples};

    use super::*;
    use crate::entropy::encode::test_support::Lcg;
    use crate::output_format::color::convert;

    #[test]
    fn source_error_mapping_keeps_invalid_internal_planes_distinct_from_source_values() {
        let cases = [
            (
                OutputFormatError::UnsupportedCombination {
                    combination: "test",
                },
                JxrErrorKind::Unsupported,
            ),
            (
                OutputFormatError::ComponentCount {
                    expected: 3,
                    actual: 2,
                },
                JxrErrorKind::InternalInvariant,
            ),
            (
                OutputFormatError::InvalidPlane {
                    component: Some(0),
                    reason: "test",
                },
                JxrErrorKind::InternalInvariant,
            ),
            (
                OutputFormatError::CropOutsidePlane { component: Some(0) },
                JxrErrorKind::InternalInvariant,
            ),
            (
                OutputFormatError::AlphaMismatch,
                JxrErrorKind::InternalInvariant,
            ),
            (
                OutputFormatError::ArithmeticOverflow { operation: "test" },
                JxrErrorKind::ArithmeticOverflow,
            ),
            (
                OutputFormatError::InvalidFloatingPointSample,
                JxrErrorKind::InvalidRequest,
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(source_error(&error).kind, expected);
        }
    }

    #[test]
    fn cmyk_and_n_component_formats_resolve_to_their_coded_colour() {
        let request = EncodeRequest::lossless();
        let direct = request.clone().with_cmyk_direct(true);
        let cmyk = PixelFormat::U8(ChannelLayout::Cmyk);
        let transformed = SourceFormat::resolve(cmyk, false, None, &request).unwrap();
        assert_eq!(
            (
                transformed.output_color,
                transformed.internal_color,
                transformed.components,
                transformed.output_color_code,
                transformed.annex_a_guid[15],
            ),
            (ColorFormat::Cmyk, ColorFormat::YuvK, 4, 4, 0x1c)
        );
        let untransformed = SourceFormat::resolve(cmyk, false, None, &direct).unwrap();
        assert_eq!(
            (
                untransformed.output_color,
                untransformed.internal_color,
                untransformed.output_color_code,
                untransformed.annex_a_guid[15],
            ),
            (ColorFormat::CmykDirect, ColorFormat::YuvK, 5, 0x54)
        );
        let deep = PixelFormat::U16(ChannelLayout::Cmyk);
        let deep = SourceFormat::resolve(deep, false, None, &request).unwrap();
        assert_eq!(deep.bit_depth, OutputBitDepth::U16 { shift_bits: 0 });
        for channels in 3..=8_u16 {
            let format = PixelFormat::U16(ChannelLayout::NComponent(channels));
            // The direct-CMYK choice does not reach other formats.
            let resolved = SourceFormat::resolve(format, false, None, &direct).unwrap();
            let color = ColorFormat::NComponent(channels);
            assert_eq!(
                (
                    resolved.output_color,
                    resolved.internal_color,
                    resolved.components,
                    resolved.output_color_code,
                    resolved.pixel_samples,
                ),
                (color, color, channels, 6, usize::from(channels))
            );
            assert_eq!(
                resolved,
                SourceFormat::resolve(format, false, None, &request).unwrap()
            );
        }
    }

    /// The output channels a decoder forms from one pixel's coded components.
    fn decoded_channels(format: SourceFormat, internal: &[i32], scaled: bool) -> Vec<i32> {
        let channels = if is_direct(format.internal_color, format.output_color) {
            internal.to_vec()
        } else {
            convert(
                format.internal_color,
                format.output_color,
                internal,
                false,
                false,
            )
            .unwrap()
            .to_vec()
        };
        (0..usize::from(format.components))
            .map(|channel| {
                format
                    .scale(channel, scaled)
                    .apply(channels[channel])
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn component_planes_invert_the_decoder_output_stage() {
        let (width, height) = (3_u32, 2_u32);
        let request = EncodeRequest::lossless();
        let direct = request.clone().with_cmyk_direct(true);
        let layout = ImageLayout::new(width, height, &request).unwrap();
        let (plane_width, _) = layout.plane_dimensions().unwrap();
        let cases = [
            (ChannelLayout::Cmyk, &request),
            (ChannelLayout::Cmyk, &direct),
            (ChannelLayout::NComponent(3), &request),
            (ChannelLayout::NComponent(8), &request),
        ];
        for (channel_layout, request) in cases {
            let channels = usize::from(channel_layout.channel_count());
            // Extremes, mid-range, and unrelated values in every channel.
            let samples: Vec<u16> = (0..6 * channels)
                .map(|index| match index % 5 {
                    0 => 0,
                    1 => 0xffff,
                    2 => 0x8000,
                    _ => u16::try_from(index * 7919 % 65_536).unwrap(),
                })
                .collect();
            let bytes: Vec<u8> = samples
                .iter()
                .map(|sample| sample.to_be_bytes()[0])
                .collect();
            for (pixel_format, source) in [
                (PixelFormat::U8(channel_layout), SourceSamples::U8(&bytes)),
                (
                    PixelFormat::U16(channel_layout),
                    SourceSamples::U16(&samples),
                ),
            ] {
                let image = SourceImage::new(width, height, pixel_format, source).unwrap();
                let format = SourceFormat::resolve(pixel_format, false, None, request).unwrap();
                for scaled in [false, true] {
                    let scaling = PlaneScaling {
                        colour: scaled,
                        alpha: false,
                    };
                    let planes =
                        component_planes(&image, &layout, format, scaling, PlaneSet::All).unwrap();
                    assert!(planes.alpha.is_none());
                    let planes = planes.colour;
                    assert_eq!(planes.len(), channels);
                    for pixel in 0..6 {
                        let (x, y) = (pixel % 3, pixel / 3);
                        let internal: Vec<i32> = planes
                            .iter()
                            .map(|plane| plane[y * plane_width + x])
                            .collect();
                        let expected: Vec<i32> = (0..channels)
                            .map(|channel| match source {
                                SourceSamples::U8(_) => {
                                    i32::from(bytes[pixel * channels + channel])
                                }
                                _ => i32::from(samples[pixel * channels + channel]),
                            })
                            .collect();
                        assert_eq!(
                            decoded_channels(format, &internal, scaled),
                            expected,
                            "{pixel_format:?} {:?} scaled {scaled}",
                            format.output_color
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn formats_resolve_through_the_container_table() {
        let request = EncodeRequest::lossless();
        let gray =
            SourceFormat::resolve(PixelFormat::U8(ChannelLayout::Luma), false, None, &request)
                .unwrap();
        assert_eq!(
            (gray.components, gray.internal_color),
            (1, ColorFormat::Luma)
        );
        let bgr = SourceFormat::resolve(PixelFormat::U8(ChannelLayout::Bgr), false, None, &request)
            .unwrap();
        assert_eq!(
            [0, 1, 2].map(|channel| bgr.channel_offset(channel)),
            [2, 1, 0]
        );
        assert_eq!((bgr.output_color_code, bgr.annex_a_guid[15]), (7, 0x0c));
        let fixed =
            SourceFormat::resolve(PixelFormat::I32(ChannelLayout::Luma), false, None, &request)
                .unwrap();
        assert_eq!(fixed.bit_depth, OutputBitDepth::I32 { shift_bits: 10 });
        let float = SourceFormat::resolve(
            PixelFormat::F32(ChannelLayout::Luma),
            false,
            None,
            &request.clone().with_float_precision(12, 3),
        )
        .unwrap();
        assert_eq!(
            float.bit_depth,
            OutputBitDepth::F32 {
                mantissa_length: 12,
                exponent_bias: 3
            }
        );
        assert!(
            SourceFormat::resolve(PixelFormat::Rgb565, false, None, &request)
                .unwrap()
                .is_packed()
        );
        let padded =
            SourceFormat::resolve(PixelFormat::U8(ChannelLayout::Bgrx), false, None, &request)
                .unwrap();
        assert_eq!((padded.components, padded.pixel_samples), (3, 4));
        for unsupported in [
            // No container format declares gray with alpha.
            PixelFormat::U8(ChannelLayout::LumaAlpha),
            PixelFormat::U8(ChannelLayout::NComponent(2)),
            PixelFormat::U8(ChannelLayout::NComponent(9)),
            PixelFormat::I16(ChannelLayout::Cmyk),
            // YUV is declared at the depths and samplings of Table A.6 only.
            PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs420)),
            PixelFormat::I16(ChannelLayout::Yuv(ChromaSampling::Cs422)),
            PixelFormat::F16(ChannelLayout::Yuv(ChromaSampling::Cs444)),
            PixelFormat::I16(ChannelLayout::Bgr),
        ] {
            assert_eq!(
                SourceFormat::resolve(unsupported, false, None, &request)
                    .unwrap_err()
                    .kind,
                JxrErrorKind::Unsupported
            );
        }
    }

    #[test]
    fn yuv_formats_are_coded_as_they_are_stored() {
        let request = EncodeRequest::lossless();
        // `(storage, sample bits, depth, GUID discriminator)`.
        for (format, sample_bits, bit_depth, code) in [
            (
                PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420)),
                None,
                OutputBitDepth::U8,
                0x44,
            ),
            (
                PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs422)),
                Some(10),
                OutputBitDepth::U10,
                0x46,
            ),
            (
                PixelFormat::U16(ChannelLayout::Yuva(ChromaSampling::Cs422)),
                None,
                OutputBitDepth::U16 { shift_bits: 0 },
                0x4f,
            ),
            (
                PixelFormat::I16(ChannelLayout::Yuv(ChromaSampling::Cs444)),
                None,
                OutputBitDepth::I16 { shift_bits: 0 },
                0x4b,
            ),
            (
                PixelFormat::U16(ChannelLayout::Yuva(ChromaSampling::Cs444)),
                Some(10),
                OutputBitDepth::U10,
                0x51,
            ),
        ] {
            let resolved = SourceFormat::resolve(format, false, sample_bits, &request).unwrap();
            let (PixelFormat::U8(layout) | PixelFormat::U16(layout) | PixelFormat::I16(layout)) =
                format
            else {
                unreachable!()
            };
            let (ChannelLayout::Yuv(sampling) | ChannelLayout::Yuva(sampling)) = layout else {
                unreachable!()
            };
            let colour = ColorFormat::Yuv(sampling);
            assert_eq!(
                (
                    resolved.output_color,
                    resolved.internal_color,
                    resolved.components,
                    resolved.bit_depth,
                    resolved.annex_a_guid[15],
                ),
                (colour, colour, 3, bit_depth, code),
                "{format:?}"
            );
            assert_eq!(resolved.alpha, format.has_alpha(), "{format:?}");
            // The components are the stored channels: nothing is converted
            // and nothing is filtered.
            assert_eq!(resolved.decimation(), None, "{format:?}");
            let mut pixel = [[-5_i32], [900], [0]];
            let mut rows = pixel.each_mut().map(<[i32; 1]>::as_mut_slice);
            resolved.internal_rows(&mut rows).unwrap();
            assert_eq!(pixel, [[-5], [900], [0]], "{format:?}");
            assert_eq!(
                resolved.output_color_code,
                output_color_code(AnnexAPixelFamily::Yuv(sampling))
            );
        }
        // A depth the table does not declare for the storage is refused.
        let wide_420 = PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420));
        assert_eq!(
            SourceFormat::resolve(wide_420, false, Some(10), &request)
                .unwrap_err()
                .kind,
            JxrErrorKind::Unsupported
        );
    }

    #[test]
    fn the_requested_chroma_sampling_reaches_full_resolution_chroma_only() {
        let rgb = PixelFormat::U8(ChannelLayout::Rgb);
        let yuv = |sampling| PixelFormat::U8(ChannelLayout::Yuv(sampling));
        for sampling in [
            ChromaSampling::Cs444,
            ChromaSampling::Cs422,
            ChromaSampling::Cs420,
        ] {
            let request = EncodeRequest::lossless().with_chroma_sampling(sampling);
            let decimation = (sampling != ChromaSampling::Cs444).then_some(sampling);
            // RGB of every storage, RGBE, and YUV 4:4:4 take the sampling.
            for format in [
                rgb,
                PixelFormat::U8(ChannelLayout::Bgra),
                PixelFormat::F32(ChannelLayout::Rgb),
                PixelFormat::Rgb565,
                PixelFormat::Rgbe,
                yuv(ChromaSampling::Cs444),
            ] {
                let resolved = SourceFormat::resolve(format, false, None, &request).unwrap();
                assert_eq!(
                    resolved.internal_color,
                    ColorFormat::Yuv(sampling),
                    "{format:?}"
                );
                assert_eq!(resolved.decimation(), decimation, "{format:?}");
                // What a decoder returns does not change.
                assert_eq!(
                    resolved.output_color,
                    SourceFormat::resolve(format, false, None, &EncodeRequest::lossless())
                        .unwrap()
                        .output_color
                );
            }
            // Formats with no chroma to subsample, and chroma that already
            // is, are coded as without the option.
            for format in [
                PixelFormat::U8(ChannelLayout::Luma),
                PixelFormat::U8(ChannelLayout::Cmyk),
                PixelFormat::U8(ChannelLayout::NComponent(3)),
                yuv(ChromaSampling::Cs422),
                yuv(ChromaSampling::Cs420),
            ] {
                assert_eq!(
                    SourceFormat::resolve(format, false, None, &request).unwrap(),
                    SourceFormat::resolve(format, false, None, &EncodeRequest::lossless()).unwrap(),
                    "{format:?} {sampling:?}"
                );
            }
        }
    }

    #[test]
    fn planar_sources_fill_each_plane_at_its_own_sampling() {
        let request = EncodeRequest::lossless();
        let (width, height) = (6_u32, 4_u32);
        let layout = ImageLayout::new(width, height, &request).unwrap();
        for (sampling, chroma_height) in [(ChromaSampling::Cs420, 2), (ChromaSampling::Cs422, 4)] {
            let pixel_format = PixelFormat::U8(ChannelLayout::Yuva(sampling));
            // Y, U, V and alpha planes of distinct values.
            let chroma = 3 * chroma_height;
            let samples: Vec<u8> = (0..24 + 2 * chroma + 24)
                .map(|index| u8::try_from(index * 3 % 251).unwrap())
                .collect();
            let image =
                SourceImage::new(width, height, pixel_format, SourceSamples::U8(&samples)).unwrap();
            let format = SourceFormat::resolve(pixel_format, false, None, &request).unwrap();
            for scaled in [false, true] {
                let scaling = PlaneScaling {
                    colour: scaled,
                    alpha: !scaled,
                };
                let planes =
                    component_planes(&image, &layout, format, scaling, PlaneSet::All).unwrap();
                // One macroblock: sixteen luma columns and eight chroma ones.
                let chroma_rows = if sampling == ChromaSampling::Cs420 {
                    8
                } else {
                    16
                };
                let sizes: Vec<usize> = planes.colour.iter().map(Vec::len).collect();
                assert_eq!(
                    sizes,
                    [256, 8 * chroma_rows, 8 * chroma_rows],
                    "{sampling:?}"
                );
                let alpha = planes.alpha.unwrap();
                assert_eq!(alpha.len(), 256);
                // `(plane, first source sample, width, height, padded width)`.
                let layouts = [
                    (&planes.colour[0], 0, 6, 4, 16),
                    (&planes.colour[1], 24, 3, chroma_height, 8),
                    (&planes.colour[2], 24 + chroma, 3, chroma_height, 8),
                    (&alpha, 24 + 2 * chroma, 6, 4, 16),
                ];
                for (index, (plane, offset, plane_width, plane_height, padded)) in
                    layouts.into_iter().enumerate()
                {
                    let scale = if index == 3 {
                        ChannelScale::alpha(format.bit_depth, !scaled)
                    } else {
                        format.scale(index, scaled)
                    };
                    for y in 0..plane.len() / padded {
                        for x in 0..padded {
                            // Positions beyond the plane repeat its edge.
                            let source = offset
                                + y.min(plane_height - 1) * plane_width
                                + x.min(plane_width - 1);
                            assert_eq!(
                                scale.apply(plane[y * padded + x]).unwrap(),
                                i32::from(samples[source]),
                                "{sampling:?} plane {index} at {x},{y}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn subsampled_coding_filters_the_padded_chroma_planes() {
        let (width, height) = (20_u32, 9_u32);
        // Red and blue a constant away from green: constant chroma.
        let pixels: Vec<u8> = (0..width * height)
            .flat_map(|index| {
                let green = u8::try_from(40 + index % 150).unwrap();
                [green + 20, green, green - 30]
            })
            .collect();
        let rgb = PixelFormat::U8(ChannelLayout::Rgb);
        let image = SourceImage::new(width, height, rgb, SourceSamples::U8(&pixels)).unwrap();
        let scaling = PlaneScaling {
            colour: true,
            alpha: false,
        };
        let lossless = EncodeRequest::lossless();
        let layout = ImageLayout::new(width, height, &lossless).unwrap();
        let full_format = SourceFormat::resolve(rgb, false, None, &lossless).unwrap();
        let full = component_planes(&image, &layout, full_format, scaling, PlaneSet::All)
            .unwrap()
            .colour;
        for (sampling, chroma_samples) in [
            (ChromaSampling::Cs422, 16 * 16),
            (ChromaSampling::Cs420, 16 * 8),
        ] {
            let request = lossless.clone().with_chroma_sampling(sampling);
            let format = SourceFormat::resolve(rgb, false, None, &request).unwrap();
            let planes = component_planes(&image, &layout, format, scaling, PlaneSet::All)
                .unwrap()
                .colour;
            // Luma is untouched; chroma has half the columns of the padded
            // plane, and for 4:2:0 half its rows.
            assert_eq!(planes[0], full[0], "{sampling:?}");
            for component in 1..3 {
                assert_eq!(planes[component].len(), chroma_samples, "{sampling:?}");
                // A constant plane stays constant through the filter.
                assert!(
                    planes[component]
                        .iter()
                        .all(|&sample| sample == full[component][0]),
                    "{sampling:?} component {component}"
                );
            }
        }
    }

    #[test]
    fn samples_beyond_the_declared_depth_are_rejected_where_they_are_used() {
        let format = PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs444));
        // Two rows of two pixels with one unused sample between the rows.
        let mut samples = [1023_u16; 13];
        samples[6] = 4000;
        let image = SourceImage::new(2, 2, format, SourceSamples::U16(&samples))
            .unwrap()
            .with_row_stride(7)
            .unwrap()
            .with_sample_bits(10)
            .unwrap();
        // Row padding is not part of the image.
        assert!(check_sample_depth(&image).is_ok());
        samples[9] = 1024;
        let image = SourceImage::new(2, 2, format, SourceSamples::U16(&samples))
            .unwrap()
            .with_row_stride(7)
            .unwrap();
        // The storage's own depth has no narrower limit.
        assert!(check_sample_depth(&image).is_ok());
        assert_eq!(
            check_sample_depth(&image.with_sample_bits(10).unwrap())
                .unwrap_err()
                .kind,
            JxrErrorKind::InvalidRequest
        );
        // A planar image is checked plane by plane.
        let planar = PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs422));
        let mut samples = [7_u16; 8];
        let within = |samples: &[u16]| {
            let image = SourceImage::new(2, 2, planar, SourceSamples::U16(samples))
                .unwrap()
                .with_sample_bits(10)
                .unwrap();
            check_sample_depth(&image).is_ok()
        };
        assert!(within(&samples));
        samples[7] = 2000;
        assert!(!within(&samples));
    }

    #[test]
    fn alpha_formats_resolve_to_their_colour_components_and_declaration() {
        let request = EncodeRequest::lossless();
        // `(format, coded components, samples per pixel, GUID discriminator)`.
        for (format, components, pixel_samples, code) in [
            (PixelFormat::U8(ChannelLayout::Bgra), 3, 4, 0x0f),
            (PixelFormat::U16(ChannelLayout::Rgba), 3, 4, 0x16),
            (PixelFormat::F16(ChannelLayout::Rgba), 3, 4, 0x3a),
            (PixelFormat::F32(ChannelLayout::Rgba), 3, 4, 0x19),
            (PixelFormat::U8(ChannelLayout::Cmyka), 4, 5, 0x2c),
            (
                PixelFormat::U16(ChannelLayout::NComponentAlpha(3)),
                3,
                4,
                0x34,
            ),
            (
                PixelFormat::U8(ChannelLayout::NComponentAlpha(8)),
                8,
                9,
                0x33,
            ),
        ] {
            let resolved = SourceFormat::resolve(format, false, None, &request).unwrap();
            assert!(
                resolved.alpha && !resolved.premultiplied_alpha,
                "{format:?}"
            );
            assert_eq!(
                (
                    resolved.components,
                    resolved.pixel_samples,
                    resolved.annex_a_guid[15]
                ),
                (components, pixel_samples, code),
                "{format:?}"
            );
        }
        let premultiplied =
            SourceFormat::resolve(PixelFormat::U8(ChannelLayout::Bgra), true, None, &request)
                .unwrap();
        assert!(premultiplied.premultiplied_alpha);
        assert_eq!(premultiplied.annex_a_guid[15], 0x10);
        // Premultiplied samples need a container format that says so.
        for undeclared in [
            PixelFormat::F16(ChannelLayout::Rgba),
            PixelFormat::U8(ChannelLayout::Cmyka),
            PixelFormat::U8(ChannelLayout::NComponentAlpha(3)),
        ] {
            assert_eq!(
                SourceFormat::resolve(undeclared, true, None, &request)
                    .unwrap_err()
                    .kind,
                JxrErrorKind::Unsupported,
                "{undeclared:?}"
            );
        }
    }

    #[test]
    fn the_alpha_plane_inverts_the_scale_a_decoder_packs_alpha_with() {
        let request = EncodeRequest::lossless();
        let layout = ImageLayout::new(2, 1, &request).unwrap();
        let samples = [10_u16, 0, 0xffff, 0x8000, 7, 300, 0x1234, 0xfffe];
        for (pixel_format, alpha_offset) in [
            (PixelFormat::U16(ChannelLayout::Rgba), 3),
            (PixelFormat::U16(ChannelLayout::NComponentAlpha(3)), 3),
        ] {
            let image = SourceImage::new(2, 1, pixel_format, SourceSamples::U16(&samples)).unwrap();
            let format = SourceFormat::resolve(pixel_format, false, None, &request).unwrap();
            // The alpha plane follows its own scaled flag, not the colour's.
            for (colour, alpha) in [(false, false), (true, false), (false, true), (true, true)] {
                let scaling = PlaneScaling { colour, alpha };
                let planes =
                    component_planes(&image, &layout, format, scaling, PlaneSet::All).unwrap();
                assert_eq!(planes.colour.len(), 3);
                let plane = planes.alpha.unwrap();
                let scale = ChannelScale::alpha(format.bit_depth, alpha);
                for pixel in 0..2 {
                    assert_eq!(
                        scale.apply(plane[pixel]).unwrap(),
                        i32::from(samples[pixel * 4 + alpha_offset]),
                        "{pixel_format:?} colour {colour} alpha {alpha}"
                    );
                }
                // Padding repeats the last pixel's alpha.
                assert_eq!(plane[2], plane[1]);
            }
        }
    }

    /// Deterministic 32-bit values for building samples of any storage.
    fn noise(count: usize, seed: u64) -> impl Iterator<Item = u32> {
        let mut state = seed;
        (0..count).map(move |_| {
            state = Lcg::advance(state);
            u32::try_from(state >> 32).unwrap()
        })
    }

    /// `count` unrelated storage elements of `format`, each within
    /// `sample_bits` when its depth is narrower than its storage.
    fn storage(
        format: PixelFormat,
        sample_bits: Option<u8>,
        count: usize,
        seed: u64,
    ) -> DecodedSamples {
        let values = noise(count, seed);
        let low16 = |value: u32| u16::try_from(value & 0xffff).unwrap();
        let mask = sample_bits.map_or(0xffff, |bits| (1_u32 << bits) - 1);
        match format {
            PixelFormat::BitPacked(_) => {
                DecodedSamples::BitPacked(values.map(|value| value.to_le_bytes()[0]).collect())
            }
            PixelFormat::U8(_) => {
                DecodedSamples::U8(values.map(|value| value.to_le_bytes()[0]).collect())
            }
            PixelFormat::U16(_) => {
                DecodedSamples::U16(values.map(|value| low16(value & mask)).collect())
            }
            PixelFormat::I16(_) => {
                DecodedSamples::I16(values.map(|value| low16(value).cast_signed()).collect())
            }
            // Within the range every shift scales without overflow.
            PixelFormat::I32(_) => {
                DecodedSamples::I32(values.map(|value| value.cast_signed() >> 6).collect())
            }
            PixelFormat::F16(_) => DecodedSamples::F16(values.map(low16).collect()),
            // Every bit pattern: denormals, infinities and NaNs among them.
            PixelFormat::F32(_) => DecodedSamples::F32(values.map(f32::from_bits).collect()),
            PixelFormat::Rgb555 => DecodedSamples::Rgb555(values.map(low16).collect()),
            PixelFormat::Rgb565 => DecodedSamples::Rgb565(values.map(low16).collect()),
            PixelFormat::Rgb101010 => DecodedSamples::Rgb101010(values.collect()),
            PixelFormat::Rgbe => DecodedSamples::Rgbe(values.collect()),
        }
    }

    /// The image of `samples`, tightly packed or, when `loose`, with unused
    /// samples after every row and its planes stored last to first.
    fn located(
        (width, height): (u32, u32),
        (format, sample_bits): (PixelFormat, Option<u8>),
        samples: &DecodedSamples,
        loose: bool,
    ) -> SourceImage<'_> {
        let image = SourceImage::new(width, height, format, SourceSamples::from(samples)).unwrap();
        let image = match sample_bits {
            Some(bits) => image.with_sample_bits(bits).unwrap(),
            None => image,
        };
        if !loose {
            return image;
        }
        let Some(count) = image.plane_count() else {
            return image.with_row_stride(image.row_stride + 3).unwrap();
        };
        let mut planes = [SourcePlane::default(); 4];
        let mut offset = 5;
        for index in (0..count).rev() {
            let (plane_width, plane_height) = image.plane_dimensions(index);
            let row_stride = usize::try_from(plane_width).unwrap() + 2;
            planes[index] = SourcePlane { offset, row_stride };
            offset += row_stride * usize::try_from(plane_height).unwrap();
        }
        image.with_planes(&planes[..count]).unwrap()
    }

    /// FNV-1a over what a conversion returned: the length and samples of
    /// every plane, or the kind of its error.
    fn fold(hash: u64, result: &Result<SourcePlanes, JxrError>) -> u64 {
        let mut hash = hash;
        let mut push = |bytes: &[u8]| {
            for &byte in bytes {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
        };
        match result {
            Ok(planes) => {
                push(&[u8::try_from(planes.colour.len()).unwrap()]);
                for plane in planes.colour.iter().chain(&planes.alpha) {
                    push(&u64::try_from(plane.len()).unwrap().to_le_bytes());
                    for sample in plane {
                        push(&sample.to_le_bytes());
                    }
                }
            }
            Err(error) => push(format!("{:?}", error.kind).as_bytes()),
        }
        hash
    }

    /// Unrelated samples for an image of `format` and the given size, with
    /// room for unused samples after its rows and planes.
    fn image_samples(
        (format, sample_bits): (PixelFormat, Option<u8>),
        (width, height): (u32, u32),
        seed: u64,
    ) -> DecodedSamples {
        let channels = usize::from(format.channel_count());
        let (width, height) = (
            usize::try_from(width).unwrap(),
            usize::try_from(height).unwrap(),
        );
        let count = (width + 3) * height * (channels + 1) + 8;
        storage(format, sample_bits, count, seed)
    }

    /// Fold the planes `image` converts to under `request` into `hash`.
    fn fold_image(
        hash: u64,
        image: &SourceImage<'_>,
        request: &EncodeRequest,
        scaling: PlaneScaling,
    ) -> u64 {
        let layout = ImageLayout::new(image.width, image.height, request).unwrap();
        let format =
            SourceFormat::resolve(image.format, false, image.sample_bits, request).unwrap();
        let together = component_planes(image, &layout, format, scaling, PlaneSet::All);
        // The colour planes and the alpha plane read apart, as the encoder
        // reads them, are the planes of one reading.
        let colour = component_planes(image, &layout, format, scaling, PlaneSet::Colour);
        let alpha = component_planes(image, &layout, format, scaling, PlaneSet::Alpha);
        match (&together, colour, alpha) {
            (Ok(together), Ok(colour), Ok(alpha)) => {
                assert!(colour.colour == together.colour && colour.alpha.is_none());
                assert!(alpha.alpha == together.alpha && alpha.colour.is_empty());
            }
            (Err(_), colour, alpha) => assert!(colour.is_err() || alpha.is_err()),
            (Ok(_), ..) => panic!("{:?} fails only when read apart", image.format),
        }
        fold(hash, &together)
    }

    /// Samples in the largest pinned image, whose rows are read as several
    /// tasks. The digests pin the number, so it is not derived from the task
    /// size.
    const PARALLEL_SAMPLES: usize = 128 * 1024;
    const _: () = assert!(PARALLEL_SAMPLES >= 2 * SAMPLES_PER_TASK);

    /// The planes of images of `format` in every size class, row layout,
    /// arithmetic, and request option that reaches the conversion, as one
    /// digest.
    fn planes_digest(format: PixelFormat, sample_bits: Option<u8>) -> u64 {
        let requests = [
            EncodeRequest::lossless(),
            EncodeRequest::lossless()
                .with_cmyk_direct(true)
                .with_chroma_sampling(ChromaSampling::Cs422)
                .with_shift_bits(0)
                .with_float_precision(12, 3),
            // Floats this wide overflow the colour transform and the scaling.
            EncodeRequest::lossless()
                .with_chroma_sampling(ChromaSampling::Cs420)
                .with_shift_bits(4)
                .with_float_precision(23, 4),
        ];
        let alpha_scalings: &[bool] = if format.has_alpha() {
            &[false, true]
        } else {
            &[false]
        };
        let format = (format, sample_bits);
        let mut hash = 0xcbf2_9ce4_8422_2325;
        // One pixel, partial macroblocks, whole ones, and several of each;
        // even, as subsampled planes need.
        for (seed, dimensions) in [(2, 2), (6, 4), (16, 16), (18, 34), (40, 24)]
            .into_iter()
            .enumerate()
        {
            let samples = image_samples(format, dimensions, u64::try_from(seed).unwrap() + 1);
            for request in &requests {
                for loose in [false, true] {
                    let image = located(dimensions, format, &samples, loose);
                    for colour in [false, true] {
                        for &alpha in alpha_scalings {
                            let scaling = PlaneScaling { colour, alpha };
                            hash = fold_image(hash, &image, request, scaling);
                        }
                    }
                }
            }
        }
        // An image whose rows are read as several parallel tasks, which
        // takes `PARALLEL_SAMPLES` in the planes that are filled together:
        // all of an interleaved image and one of a planar image.
        let together = match format.0 {
            PixelFormat::U8(layout) | PixelFormat::U16(layout) => match layout {
                ChannelLayout::Yuv(ChromaSampling::Cs422 | ChromaSampling::Cs420)
                | ChannelLayout::Yuva(ChromaSampling::Cs422 | ChromaSampling::Cs420) => 1,
                _ => layout.channel_count(),
            },
            other => other.channel_count(),
        };
        let height = PARALLEL_SAMPLES.div_ceil(400 * usize::from(together)) + 2;
        let dimensions = (400, u32::try_from(height).unwrap().next_multiple_of(2));
        let samples = image_samples(format, dimensions, 6);
        let image = located(dimensions, format, &samples, true);
        let scaling = PlaneScaling {
            colour: true,
            alpha: false,
        };
        hash = fold_image(hash, &image, &requests[1], scaling);
        hash
    }

    #[test]
    fn a_sample_that_cannot_be_coded_fails_the_conversion_from_any_row() {
        let request = EncodeRequest::lossless().with_shift_bits(0);
        let format = PixelFormat::I32(ChannelLayout::Rgb);
        let resolved = SourceFormat::resolve(format, false, None, &request).unwrap();
        let scaling = PlaneScaling {
            colour: true,
            alpha: false,
        };
        // Rows read as one task, and as several.
        for (width, height) in [(8_u32, 4_u32), (400, 300)] {
            let layout = ImageLayout::new(width, height, &request).unwrap();
            let count = usize::try_from(width * height * 3).unwrap();
            let convert = |samples: &[i32]| {
                let image =
                    SourceImage::new(width, height, format, SourceSamples::I32(samples)).unwrap();
                component_planes(&image, &layout, resolved, scaling, PlaneSet::All)
                    .map(|planes| planes.colour)
            };
            let mut samples = vec![7_i32; count];
            assert_eq!(convert(&samples).unwrap().len(), 3);
            for position in [0, count / 2, count - 1] {
                // Scaled arithmetic has no room for the sample's three
                // fractional bits.
                samples[position] = i32::MAX;
                assert_eq!(
                    convert(&samples).unwrap_err().kind,
                    JxrErrorKind::ArithmeticOverflow,
                    "{width}x{height} sample {position}"
                );
                samples[position] = 7;
            }
        }
    }

    /// Every format of one sample storage that the container table declares.
    fn declared_formats(
        storage: fn(ChannelLayout) -> PixelFormat,
    ) -> Vec<(PixelFormat, Option<u8>)> {
        let samplings = [
            ChromaSampling::Cs444,
            ChromaSampling::Cs422,
            ChromaSampling::Cs420,
        ];
        let mut layouts = vec![
            ChannelLayout::Luma,
            ChannelLayout::Rgb,
            ChannelLayout::Rgbx,
            ChannelLayout::Rgba,
            ChannelLayout::Bgr,
            ChannelLayout::Bgrx,
            ChannelLayout::Bgra,
            ChannelLayout::Cmyk,
            ChannelLayout::Cmyka,
        ];
        layouts.extend(samplings.map(ChannelLayout::Yuv));
        layouts.extend(samplings.map(ChannelLayout::Yuva));
        layouts.extend((3..=8).map(ChannelLayout::NComponent));
        layouts.extend((3..=8).map(ChannelLayout::NComponentAlpha));
        let request = EncodeRequest::lossless();
        layouts
            .into_iter()
            .flat_map(|layout| [(storage(layout), None), (storage(layout), Some(10))])
            .filter(|&(format, bits)| SourceFormat::resolve(format, false, bits, &request).is_ok())
            .collect()
    }

    /// Guards the conversion's output through every restructuring of it: the
    /// digests were taken from the one-sample-at-a-time implementation. A
    /// digest that changes means planes changed, and with them every file
    /// encoded from that storage.
    #[test]
    fn planes_of_every_storage_and_layout_are_pinned() {
        let packed = |format| vec![(format, None)];
        // `(storage, formats declared for it, digest of their planes)`.
        let pinned = [
            (
                "1-bit",
                declared_formats(PixelFormat::BitPacked),
                1,
                0x7df4_4f42_3823_e859_u64,
            ),
            (
                "u8",
                declared_formats(PixelFormat::U8),
                26,
                0xc78b_f31a_2985_4a45,
            ),
            (
                "u16",
                declared_formats(PixelFormat::U16),
                25,
                0xd8ca_5ceb_4f25_2e9d,
            ),
            (
                "i16",
                declared_formats(PixelFormat::I16),
                6,
                0x0b32_6559_e80d_80c8,
            ),
            (
                "i32",
                declared_formats(PixelFormat::I32),
                4,
                0x28a2_0fae_d009_2355,
            ),
            (
                "f16",
                declared_formats(PixelFormat::F16),
                4,
                0x8e85_be1a_2fef_dee9,
            ),
            (
                "f32",
                declared_formats(PixelFormat::F32),
                4,
                0x29c2_7c6d_ebb5_65e2,
            ),
            (
                "rgb555",
                packed(PixelFormat::Rgb555),
                1,
                0x1165_776c_a05e_6bfd,
            ),
            (
                "rgb565",
                packed(PixelFormat::Rgb565),
                1,
                0xe331_ec22_dd77_dedb,
            ),
            (
                "rgb101010",
                packed(PixelFormat::Rgb101010),
                1,
                0xb020_70c1_1728_d332,
            ),
            ("rgbe", packed(PixelFormat::Rgbe), 1, 0x8649_304c_db66_74e7),
        ];
        for (storage, formats, count, digest) in pinned {
            assert_eq!(formats.len(), count, "{storage}");
            let actual = formats
                .into_iter()
                .fold(0_u64, |hash, (format, sample_bits)| {
                    hash.rotate_left(7) ^ planes_digest(format, sample_bits)
                });
            assert_eq!(actual, digest, "{storage}: {actual:#018x}");
        }
    }
}
