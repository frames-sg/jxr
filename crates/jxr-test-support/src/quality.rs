// SPDX-License-Identifier: MIT OR Apache-2.0

//! Source classes of the encoder's quality scale.
//!
//! The scale promises that one quality means the same for a picture however
//! it is stored. A [`QualityClass`] is one way of storing it: a sample type,
//! a colour format, or an encoder option that changes what is coded. Each
//! class carries the same 8-bit [`QualityPicture`] and measures a decoded
//! image against it on one scale, in units of an 8-bit sample value, so that
//! sizes and errors compare across classes.

use jxr::{
    ChannelLayout, ChromaSampling, DecodeRequest, DecodedSamples, EncodeRequest, JxrView,
    PixelFormat, SourceImage, SourceSamples,
};

use crate::oracle_format;

/// An 8-bit picture the scale is measured on: gray, or RGB with its samples
/// interleaved. Rows are packed end to end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualityPicture {
    /// Name the picture is reported under.
    pub name: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Samples of a pixel: one for gray, three for RGB.
    pub channels: usize,
    /// The samples, `width * height * channels` of them.
    pub pixels: Vec<u8>,
}

/// One way of storing a picture that the quality scale is calibrated for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityClass {
    /// 8-bit gray.
    Gray8,
    /// 8-bit RGB with full-resolution chroma: the class the scale was first
    /// calibrated on.
    Rgb8,
    /// 8-bit RGB whose chroma the encoder subsamples to 4:2:2.
    Rgb8Chroma422,
    /// 8-bit RGB whose chroma the encoder subsamples to 4:2:0.
    Rgb8Chroma420,
    /// 8-bit RGB with an alpha channel.
    Rgba8,
    /// 16-bit gray.
    Gray16,
    /// 16-bit RGB.
    Rgb16,
    /// 16-bit fixed-point RGB, s2.13.
    RgbFixed16,
    /// 32-bit fixed-point RGB, s7.24.
    RgbFixed32,
    /// Half-float RGB holding linear light.
    RgbHalf,
    /// Half-float RGB holding linear light, with an alpha channel.
    RgbaHalf,
    /// 32-bit float RGB holding linear light.
    RgbFloat,
    /// Radiance RGBE holding linear light.
    Rgbe,
    /// Packed 10-bit RGB.
    Rgb101010,
    /// Packed 5:6:5 RGB.
    Rgb565,
    /// Packed 5:5:5 RGB.
    Rgb555,
    /// One bit per pixel, black or white.
    BlackWhite,
    /// 8-bit CMYK, coded through the colour transform.
    Cmyk8,
    /// 8-bit CMYK, coded channel by channel as `CMYKDIRECT`.
    CmykDirect8,
    /// 16-bit CMYK, coded through the colour transform.
    Cmyk16,
    /// Four 8-bit channels coded as they are: red, green, blue and luma.
    Channels8,
    /// 8-bit YUV 4:4:4 source samples.
    Yuv444,
    /// 8-bit YUV 4:2:2 source samples.
    Yuv422,
    /// 8-bit YUV 4:2:0 source samples.
    Yuv420,
}

/// How the samples of a class map to the common scale and to pixels.
#[derive(Clone, Copy, PartialEq)]
enum Samples {
    /// Integers that span `peak`, `channels` to a pixel.
    Linear { peak: f64, channels: usize },
    /// Linear-light floats, three to a pixel, and then a linear alpha
    /// sample where `alpha` is set.
    Light { alpha: bool },
    /// One packed word per pixel with fields of the given widths.
    Packed([u32; 3]),
    /// One RGBE word per pixel.
    Rgbe,
    /// One bit per pixel, rows padded to bytes.
    Bits,
}

impl QualityClass {
    /// Every class, in the order the tables list them.
    pub const ALL: [Self; 24] = [
        Self::Gray8,
        Self::Rgb8,
        Self::Rgb8Chroma422,
        Self::Rgb8Chroma420,
        Self::Rgba8,
        Self::Gray16,
        Self::Rgb16,
        Self::RgbFixed16,
        Self::RgbFixed32,
        Self::RgbHalf,
        Self::RgbaHalf,
        Self::RgbFloat,
        Self::Rgbe,
        Self::Rgb101010,
        Self::Rgb565,
        Self::Rgb555,
        Self::BlackWhite,
        Self::Cmyk8,
        Self::CmykDirect8,
        Self::Cmyk16,
        Self::Channels8,
        Self::Yuv444,
        Self::Yuv422,
        Self::Yuv420,
    ];

    /// Name of the class on the command line and in reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gray8 => "gray8",
            Self::Rgb8 => "rgb8",
            Self::Rgb8Chroma422 => "rgb8-422",
            Self::Rgb8Chroma420 => "rgb8-420",
            Self::Rgba8 => "rgba8",
            Self::Gray16 => "gray16",
            Self::Rgb16 => "rgb16",
            Self::RgbFixed16 => "rgb-fixed16",
            Self::RgbFixed32 => "rgb-fixed32",
            Self::RgbHalf => "rgb-half",
            Self::RgbaHalf => "rgba-half",
            Self::RgbFloat => "rgb-float",
            Self::Rgbe => "rgbe",
            Self::Rgb101010 => "rgb101010",
            Self::Rgb565 => "rgb565",
            Self::Rgb555 => "rgb555",
            Self::BlackWhite => "bw1",
            Self::Cmyk8 => "cmyk8",
            Self::CmykDirect8 => "cmyk8-direct",
            Self::Cmyk16 => "cmyk16",
            Self::Channels8 => "channels8",
            Self::Yuv444 => "yuv444",
            Self::Yuv422 => "yuv422",
            Self::Yuv420 => "yuv420",
        }
    }

    /// The class of that name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.name() == name)
    }

    /// How the class stores the picture and what its error is measured
    /// against: the peak that an error of 255 on the common scale stands for.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Gray8 | Self::Rgb8 => "the picture as it is; peak 255",
            Self::Rgb8Chroma422 | Self::Rgb8Chroma420 => {
                "the picture, chroma subsampled by the encoder; peak 255 on the RGB samples"
            }
            Self::Rgba8 => {
                "the picture with an alpha channel made from its luma, (luma - 64) * 2 clipped; \
                 peak 255, colour and alpha measured apart"
            }
            Self::Gray16 | Self::Rgb16 => "each sample times 257 plus low-order detail; peak 65535",
            Self::RgbFixed16 => "the 16-bit samples scaled so that 65535 is 1.0; peak 1.0 (8192)",
            Self::RgbFixed32 => {
                "the 16-bit samples scaled so that 65535 is 1.0; peak 1.0 (16777216)"
            }
            Self::RgbHalf | Self::RgbFloat | Self::Rgbe => {
                "linear light of the 16-bit samples read as sRGB; error measured after the sRGB \
                 transfer function, peak 1.0"
            }
            Self::RgbaHalf => {
                "the same linear light with the alpha channel of rgba8 as a float; colour as \
                 rgb-half, alpha against 1.0"
            }
            Self::Rgb101010 => "the 16-bit samples rounded to 10 bits; peak 1023",
            Self::Rgb565 => "the 16-bit samples rounded to 5, 6 and 5 bits; peak 31, 63 and 31",
            Self::Rgb555 => "the 16-bit samples rounded to 5 bits; peak 31",
            Self::BlackWhite => "the picture dithered to black and white; peak 1",
            Self::Cmyk8 | Self::CmykDirect8 => {
                "inks 255 - R, 255 - G, 255 - B with their minimum moved to black; peak 255 on \
                 the four inks"
            }
            Self::Cmyk16 => "the same inks from the 16-bit samples; peak 65535 on the four inks",
            Self::Channels8 => "red, green, blue and Rec. 601 luma as four channels; peak 255",
            Self::Yuv444 | Self::Yuv422 | Self::Yuv420 => {
                "full-range Rec. 601 YCbCr, chroma averaged where subsampled; peak 255 on the \
                 stored Y, Cb and Cr samples"
            }
        }
    }

    /// Whether the class is made from a gray picture rather than an RGB one.
    #[must_use]
    pub const fn is_gray(self) -> bool {
        matches!(self, Self::Gray8 | Self::Gray16 | Self::BlackWhite)
    }

    const fn format(self) -> PixelFormat {
        match self {
            Self::Gray8 => PixelFormat::U8(ChannelLayout::Luma),
            Self::Rgb8 | Self::Rgb8Chroma422 | Self::Rgb8Chroma420 => {
                PixelFormat::U8(ChannelLayout::Rgb)
            }
            Self::Rgba8 => PixelFormat::U8(ChannelLayout::Rgba),
            Self::Gray16 => PixelFormat::U16(ChannelLayout::Luma),
            Self::Rgb16 => PixelFormat::U16(ChannelLayout::Rgb),
            Self::RgbFixed16 => PixelFormat::I16(ChannelLayout::Rgb),
            Self::RgbFixed32 => PixelFormat::I32(ChannelLayout::Rgb),
            Self::RgbHalf => PixelFormat::F16(ChannelLayout::Rgb),
            Self::RgbaHalf => PixelFormat::F16(ChannelLayout::Rgba),
            Self::RgbFloat => PixelFormat::F32(ChannelLayout::Rgb),
            Self::Rgbe => PixelFormat::Rgbe,
            Self::Rgb101010 => PixelFormat::Rgb101010,
            Self::Rgb565 => PixelFormat::Rgb565,
            Self::Rgb555 => PixelFormat::Rgb555,
            Self::BlackWhite => PixelFormat::BitPacked(ChannelLayout::Luma),
            Self::Cmyk8 | Self::CmykDirect8 => PixelFormat::U8(ChannelLayout::Cmyk),
            Self::Cmyk16 => PixelFormat::U16(ChannelLayout::Cmyk),
            Self::Channels8 => PixelFormat::U8(ChannelLayout::NComponent(4)),
            Self::Yuv444 => PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs444)),
            Self::Yuv422 => PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs422)),
            Self::Yuv420 => PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420)),
        }
    }

    const fn samples(self) -> Samples {
        const fn linear(peak: f64, channels: usize) -> Samples {
            Samples::Linear { peak, channels }
        }
        match self {
            // Planar 4:2:2 and 4:2:0 hold one channel in each plane.
            Self::Gray8 | Self::Yuv422 | Self::Yuv420 => linear(255.0, 1),
            Self::Rgb8 | Self::Rgb8Chroma422 | Self::Rgb8Chroma420 | Self::Yuv444 => {
                linear(255.0, 3)
            }
            Self::Rgba8 | Self::Cmyk8 | Self::CmykDirect8 | Self::Channels8 => linear(255.0, 4),
            Self::Gray16 => linear(65535.0, 1),
            Self::Rgb16 => linear(65535.0, 3),
            Self::Cmyk16 => linear(65535.0, 4),
            Self::RgbFixed16 => linear(8192.0, 3),
            Self::RgbFixed32 => linear(16_777_216.0, 3),
            Self::RgbHalf | Self::RgbFloat => Samples::Light { alpha: false },
            Self::RgbaHalf => Samples::Light { alpha: true },
            Self::Rgbe => Samples::Rgbe,
            Self::Rgb101010 => Samples::Packed([10, 10, 10]),
            Self::Rgb565 => Samples::Packed([5, 6, 5]),
            Self::Rgb555 => Samples::Packed([5, 5, 5]),
            Self::BlackWhite => Samples::Bits,
        }
    }

    /// Whether a pixel ends with an alpha sample.
    const fn has_alpha(self) -> bool {
        matches!(self, Self::Rgba8 | Self::RgbaHalf)
    }

    /// The chroma sampling of a planar class.
    const fn planar(self) -> Option<ChromaSampling> {
        match self {
            Self::Yuv422 => Some(ChromaSampling::Cs422),
            Self::Yuv420 => Some(ChromaSampling::Cs420),
            _ => None,
        }
    }

    /// The request that codes this class at `quality`.
    #[must_use]
    pub const fn request(self, quality: u8) -> EncodeRequest {
        let request = EncodeRequest::quality(quality);
        match self {
            Self::Rgb8Chroma422 => request.with_chroma_sampling(ChromaSampling::Cs422),
            Self::Rgb8Chroma420 => request.with_chroma_sampling(ChromaSampling::Cs420),
            Self::CmykDirect8 => request.with_cmyk_direct(true),
            _ => request,
        }
    }

    /// Store `picture` as this class, or `None` when the class is made from
    /// the other kind of picture, gray or RGB.
    #[must_use]
    pub fn source(self, picture: &QualityPicture) -> Option<QualitySource> {
        if self.is_gray() != (picture.channels == 1) {
            return None;
        }
        // Planar chroma halves an axis, which must then be even.
        let (width, height) = match self.planar() {
            Some(ChromaSampling::Cs420) => (picture.width & !1, picture.height & !1),
            Some(_) => (picture.width & !1, picture.height),
            None => (picture.width, picture.height),
        };
        if width == 0 || height == 0 {
            return None;
        }
        let rows = picture
            .pixels
            .chunks_exact(picture.width as usize * picture.channels)
            .take(height as usize)
            .flat_map(|row| &row[..width as usize * picture.channels])
            .copied()
            .collect::<Vec<u8>>();
        let samples = derive::samples(self, &rows, width as usize);
        Some(QualitySource {
            class: self,
            name: picture.name.clone(),
            width,
            height,
            reference: Normalized::new(self, &samples, width as usize, height as usize),
            samples,
        })
    }
}

/// A picture stored as one class.
#[derive(Debug, Clone)]
pub struct QualitySource {
    /// The class the picture is stored as.
    pub class: QualityClass,
    /// Name of the picture.
    pub name: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    samples: DecodedSamples,
    reference: Normalized,
}

/// What coding a source at one setting cost and lost.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityMeasurement {
    /// Size of the encoded file.
    pub bytes: usize,
    /// Sum of squared errors over the colour samples, on the 8-bit scale.
    pub squared_error: f64,
    /// Number of colour samples.
    pub samples: usize,
    /// Largest error of a colour sample, on the 8-bit scale.
    pub max_error: f64,
    /// Sum of squared luma errors and the luma SSIM, for a class with luma.
    pub luma: Option<(f64, f64)>,
    /// Sum of squared alpha errors, for a class with alpha.
    pub alpha_squared_error: Option<f64>,
    /// DC, low-pass and high-pass parameters of each coded component, and
    /// then of the alpha plane.
    pub parameters: Vec<[u8; 3]>,
    /// Overlap mode the file declares.
    pub overlap: u8,
}

impl QualityMeasurement {
    /// PSNR of the colour samples against the class's peak; infinite for an
    /// exact image.
    #[must_use]
    pub fn psnr(&self) -> f64 {
        psnr(self.squared_error, self.samples)
    }

    /// PSNR of luma, for a class with luma.
    #[must_use]
    pub fn luma_psnr(&self, pixels: usize) -> Option<f64> {
        self.luma
            .map(|(squared_error, _)| psnr(squared_error, pixels))
    }

    /// PSNR of the alpha channel, for a class with alpha.
    #[must_use]
    pub fn alpha_psnr(&self, pixels: usize) -> Option<f64> {
        self.alpha_squared_error
            .map(|squared_error| psnr(squared_error, pixels))
    }
}

/// PSNR against 255 of a summed squared error over `samples` values.
fn psnr(squared_error: f64, samples: usize) -> f64 {
    // Sample counts stay far below the 2^53 an f64 counts exactly.
    let samples = u32::try_from(samples).map_or(f64::INFINITY, f64::from);
    10.0 * (255.0 * 255.0 * samples / squared_error).log10()
}

impl QualitySource {
    /// Pixels in the image.
    #[must_use]
    pub const fn pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// The image as the encoder borrows it.
    pub fn image(&self) -> Result<SourceImage<'_>, String> {
        SourceImage::new(
            self.width,
            self.height,
            self.class.format(),
            SourceSamples::from(&self.samples),
        )
        .map_err(|error| error.to_string())
    }

    /// Encode the source as `request` asks and compare what decodes with it.
    pub fn measure(&self, request: &EncodeRequest) -> Result<QualityMeasurement, String> {
        let file = jxr::encode(&self.image()?, request).map_err(|error| error.to_string())?;
        let decoded = self.decode(&file)?;
        let coded = Normalized::new(
            self.class,
            &decoded,
            self.width as usize,
            self.height as usize,
        );
        let (colour_error, max_error) = squared_error(&self.reference.colour, &coded.colour);
        let luma = self.reference.luma.as_ref().zip(coded.luma.as_ref());
        let alpha = self.reference.alpha.as_ref().zip(coded.alpha.as_ref());
        let (parameters, overlap) = coding(&file)?;
        Ok(QualityMeasurement {
            bytes: file.len(),
            squared_error: colour_error,
            samples: self.reference.colour.len(),
            max_error,
            luma: luma.map(|(source, coded)| {
                (
                    squared_error(source, coded).0,
                    ssim(source, coded, self.width as usize, self.height as usize),
                )
            }),
            alpha_squared_error: alpha.map(|(source, coded)| squared_error(source, coded).0),
            parameters,
            overlap,
        })
    }

    /// Decode `file` to the class's format with rows and planes packed end
    /// to end, as the source holds them.
    fn decode(&self, file: &[u8]) -> Result<DecodedSamples, String> {
        let view = JxrView::parse(file).map_err(|error| error.to_string())?;
        let layout = oracle_format(view.info()).map_err(|error| error.to_string())?;
        if layout.pixel_format != self.class.format() {
            return Err(format!("encoded file declares {:?}", layout.pixel_format));
        }
        let mut request = DecodeRequest::new(layout.pixel_format);
        request.alpha = layout.alpha;
        let image = view
            .decoder()
            .decode(&request)
            .map_err(|error| error.to_string())?;
        let source = self.image()?;
        // Storage elements in each row of each plane, and its rows.
        let geometry: Vec<(usize, usize)> = match source.plane_count() {
            Some(count) => (0..count)
                .map(|plane| {
                    let (width, height) = source.plane_dimensions(plane);
                    (width as usize, height as usize)
                })
                .collect(),
            None => vec![(
                source.row_samples().map_err(|error| error.to_string())?,
                self.height as usize,
            )],
        };
        if geometry.len() != image.planes.len() {
            return Err(format!("decode returned {} planes", image.planes.len()));
        }
        let rows = |element_bytes: usize| {
            geometry
                .iter()
                .zip(&image.planes)
                .flat_map(move |(&(width, height), plane)| {
                    (0..height).map(move |row| {
                        let start =
                            (plane.byte_offset + row * plane.row_stride_bytes) / element_bytes;
                        start..start + width
                    })
                })
        };
        Ok(match &image.samples {
            DecodedSamples::BitPacked(samples) => {
                DecodedSamples::BitPacked(packed(samples, rows(1)))
            }
            DecodedSamples::U8(samples) => DecodedSamples::U8(packed(samples, rows(1))),
            DecodedSamples::U16(samples) => DecodedSamples::U16(packed(samples, rows(2))),
            DecodedSamples::I16(samples) => DecodedSamples::I16(packed(samples, rows(2))),
            DecodedSamples::I32(samples) => DecodedSamples::I32(packed(samples, rows(4))),
            DecodedSamples::F16(samples) => DecodedSamples::F16(packed(samples, rows(2))),
            DecodedSamples::F32(samples) => DecodedSamples::F32(packed(samples, rows(4))),
            DecodedSamples::Rgb555(samples) => DecodedSamples::Rgb555(packed(samples, rows(2))),
            DecodedSamples::Rgb565(samples) => DecodedSamples::Rgb565(packed(samples, rows(2))),
            DecodedSamples::Rgb101010(samples) => {
                DecodedSamples::Rgb101010(packed(samples, rows(4)))
            }
            DecodedSamples::Rgbe(samples) => DecodedSamples::Rgbe(packed(samples, rows(4))),
        })
    }
}

/// The elements of `samples` in each of `rows`, end to end.
fn packed<T: Copy>(samples: &[T], rows: impl Iterator<Item = std::ops::Range<usize>>) -> Vec<T> {
    rows.flat_map(|row| &samples[row]).copied().collect()
}

/// The parameters of every component and then of the alpha plane, and the
/// overlap mode, as `file` declares them.
fn coding(file: &[u8]) -> Result<(Vec<[u8; 3]>, u8), String> {
    let container = jxr_native::parse_annex_a(file).map_err(|error| error.to_string())?;
    let headers = |range: std::ops::Range<usize>| {
        jxr_native::parse_codestream_headers(&file[range]).map_err(|error| error.to_string())
    };
    let plane = |plane: &jxr_native::ImagePlaneHeader| {
        let band = |set: &Option<jxr_native::QuantizerSet>, component: usize| {
            set.as_ref()
                .and_then(|set| set.components.get(component))
                .copied()
        };
        (0..usize::from(plane.components))
            .map(|component| {
                let dc = band(&plane.dc_quantizers, component).unwrap_or(0);
                let low_pass = band(&plane.lp_quantizers, component).unwrap_or(dc);
                [
                    dc,
                    low_pass,
                    band(&plane.hp_quantizers, component).unwrap_or(low_pass),
                ]
            })
            .collect::<Vec<_>>()
    };
    let primary = headers(container.codestream_range.clone())?;
    let mut parameters = plane(&primary.primary);
    if let Some(alpha) = &primary.alpha {
        parameters.extend(plane(alpha));
    }
    if let Some(range) = container.alpha_range {
        parameters.extend(plane(&headers(range)?.primary));
    }
    Ok((parameters, primary.image.overlap_mode))
}

/// The samples of an image on the common scale: 255 is the class's peak.
#[derive(Debug, Clone)]
struct Normalized {
    /// Every colour sample, in storage order.
    colour: Vec<f64>,
    /// Luma of each pixel, for a class that has one.
    luma: Option<Vec<f64>>,
    /// Alpha of each pixel, for a class with an alpha channel.
    alpha: Option<Vec<f64>>,
}

impl Normalized {
    fn new(class: QualityClass, samples: &DecodedSamples, width: usize, height: usize) -> Self {
        let channels = match class.samples() {
            Samples::Linear { channels, .. } => channels,
            Samples::Bits => 1,
            Samples::Light { .. } | Samples::Packed(_) | Samples::Rgbe => 3,
        };
        let mut colour = normalize::values(class.samples(), samples, width);
        let alpha = class.has_alpha().then(|| {
            let alpha = colour.iter().skip(3).step_by(4).copied().collect();
            colour = colour
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|&[red, green, blue, _]| [red, green, blue])
                .collect();
            alpha
        });
        let luma = match class {
            QualityClass::Cmyk8
            | QualityClass::CmykDirect8
            | QualityClass::Cmyk16
            | QualityClass::Channels8 => None,
            // The Y plane comes first.
            QualityClass::Yuv422 | QualityClass::Yuv420 => Some(colour[..width * height].to_vec()),
            QualityClass::Yuv444 => Some(colour.iter().step_by(3).copied().collect()),
            _ if channels == 1 => Some(colour.clone()),
            _ => Some(
                colour
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|[red, green, blue]| 0.299 * red + 0.587 * green + 0.114 * blue)
                    .collect(),
            ),
        };
        Self {
            colour,
            luma,
            alpha,
        }
    }
}

/// Sum of squared differences and the largest difference.
fn squared_error(source: &[f64], coded: &[f64]) -> (f64, f64) {
    source
        .iter()
        .zip(coded)
        .fold((0.0, 0.0_f64), |(sum, largest), (source, coded)| {
            let error = (coded - source).abs();
            (sum + error * error, largest.max(error))
        })
}

/// Mean structural similarity over 8-by-8 windows placed every four samples.
fn ssim(source: &[f64], coded: &[f64], width: usize, height: usize) -> f64 {
    const WINDOW: usize = 8;
    const AREA: f64 = 64.0;
    // The usual stabilizers for 8-bit samples: (0.01 * 255)^2 and (0.03 * 255)^2.
    const C1: f64 = 6.5025;
    const C2: f64 = 58.5225;
    let (mut total, mut windows) = (0.0, 0.0);
    for top in (0..height.saturating_sub(WINDOW - 1)).step_by(WINDOW / 2) {
        for left in (0..width.saturating_sub(WINDOW - 1)).step_by(WINDOW / 2) {
            // Sums of each image, of their squares, and of their products.
            let (mut sums, mut squares, mut product) = ([0.0; 2], [0.0; 2], 0.0);
            for y in top..top + WINDOW {
                let row = y * width + left..y * width + left + WINDOW;
                for (&original, &decoded) in source[row.clone()].iter().zip(&coded[row]) {
                    sums[0] += original;
                    sums[1] += decoded;
                    squares[0] += original * original;
                    squares[1] += decoded * decoded;
                    product += original * decoded;
                }
            }
            let means = [sums[0] / AREA, sums[1] / AREA];
            let variances = [
                squares[0] / AREA - means[0] * means[0],
                squares[1] / AREA - means[1] * means[1],
            ];
            let covariance = product / AREA - means[0] * means[1];
            total += ((2.0 * means[0] * means[1] + C1) * (2.0 * covariance + C2))
                / ((means[0] * means[0] + means[1] * means[1] + C1)
                    * (variances[0] + variances[1] + C2));
            windows += 1.0;
        }
    }
    // An image smaller than one window has no structure to compare.
    if windows == 0.0 { 1.0 } else { total / windows }
}

/// Stored samples to the common scale.
mod normalize {
    use jxr::DecodedSamples;

    use super::{Samples, light};

    /// Every sample of an image on the scale where 255 is the class's peak.
    pub(super) fn values(kind: Samples, samples: &DecodedSamples, width: usize) -> Vec<f64> {
        match (kind, samples) {
            (Samples::Linear { peak, .. }, samples) => {
                let scale = 255.0 / peak;
                match samples {
                    DecodedSamples::U8(samples) => linear(samples, scale),
                    DecodedSamples::U16(samples) => linear(samples, scale),
                    DecodedSamples::I16(samples) => linear(samples, scale),
                    DecodedSamples::I32(samples) => linear(samples, scale),
                    _ => Vec::new(),
                }
            }
            (Samples::Light { alpha }, DecodedSamples::F16(samples)) => samples
                .iter()
                .enumerate()
                .map(|(index, &bits)| {
                    let value = f64::from(light::half_value(bits));
                    // Alpha is a coverage, already linear in what it means.
                    if alpha && index % 4 == 3 {
                        255.0 * value
                    } else {
                        light::display(value)
                    }
                })
                .collect(),
            (Samples::Light { .. }, DecodedSamples::F32(samples)) => samples
                .iter()
                .map(|&value| light::display(f64::from(value)))
                .collect(),
            (
                Samples::Packed(bits),
                DecodedSamples::Rgb555(samples) | DecodedSamples::Rgb565(samples),
            ) => samples
                .iter()
                .flat_map(|&word| fields(u32::from(word), bits))
                .collect(),
            (Samples::Packed(bits), DecodedSamples::Rgb101010(samples)) => samples
                .iter()
                .flat_map(|&word| fields(word, bits))
                .collect(),
            (Samples::Rgbe, DecodedSamples::Rgbe(samples)) => samples
                .iter()
                .flat_map(|&word| light::rgbe_values(word).map(light::display))
                .collect(),
            (Samples::Bits, DecodedSamples::BitPacked(samples)) => samples
                .chunks_exact(width.div_ceil(8))
                .flat_map(|row| {
                    (0..width).map(|x| 255.0 * f64::from(row[x / 8] >> (7 - x % 8) & 1))
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    fn linear<T: Copy + Into<f64>>(samples: &[T], scale: f64) -> Vec<f64> {
        samples
            .iter()
            .map(|&sample| sample.into() * scale)
            .collect()
    }

    /// The red, green and blue fields of a packed word, most significant
    /// first, each on the scale of its own width.
    fn fields(word: u32, bits: [u32; 3]) -> [f64; 3] {
        let mut shift = bits.iter().sum::<u32>();
        bits.map(|width| {
            shift -= width;
            let peak = (1_u32 << width) - 1;
            255.0 * f64::from(word >> shift & peak) / f64::from(peak)
        })
    }
}

/// Linear light: the sRGB transfer function and the float storage forms.
mod light {
    /// Linear light of an sRGB-encoded value in zero to one.
    pub(super) fn linear(encoded: f32) -> f32 {
        if encoded <= 0.040_45 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    }

    /// The sRGB encoding of linear light on the scale where one is 255,
    /// extended through zero as scRGB extends it.
    pub(super) fn display(linear: f64) -> f64 {
        let magnitude = linear.abs();
        let encoded = if magnitude <= 0.003_130_8 {
            12.92 * magnitude
        } else {
            1.055 * magnitude.powf(1.0 / 2.4) - 0.055
        };
        255.0 * encoded.copysign(linear)
    }

    /// The half-precision bit pattern nearest a non-negative `value`.
    pub(super) fn half_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let exponent = (bits >> 23 & 0xff).cast_signed() - 127 + 15;
        let mantissa = bits & 0x7f_ffff;
        // Mantissa in units of the half float's last place, rounded to
        // nearest; a carry out of it lands in the exponent, as it should.
        let pattern = if exponent >= 1 {
            (exponent.cast_unsigned() << 10) + ((mantissa + (1 << 12)) >> 13)
        } else if exponent >= -10 {
            // Below the smallest normal half: no implicit leading one.
            let shift = (14 - exponent).cast_unsigned();
            ((mantissa | 1 << 23) + (1 << (shift - 1))) >> shift
        } else {
            0
        };
        u16::try_from(pattern.min(0x7bff)).unwrap_or(0x7bff)
    }

    /// The value of a half-precision bit pattern.
    pub(super) fn half_value(bits: u16) -> f32 {
        let exponent = i32::from(bits >> 10 & 0x1f);
        let mantissa = f32::from(bits & 0x3ff);
        let magnitude = if exponent == 0 {
            mantissa * 2.0_f32.powi(-24)
        } else {
            (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent - 15)
        };
        if bits & 0x8000 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    /// The RGBE word of a pixel of non-negative linear light, with the
    /// largest mantissa its exponent allows.
    pub(super) fn rgbe_word(pixel: [f32; 3]) -> u32 {
        let brightest = pixel.into_iter().fold(0.0_f32, f32::max);
        // `brightest` is `fraction * 2^exponent` with the fraction in a half
        // to one, which the format stores as a mantissa of 128 to 255.
        let exponent = (brightest.to_bits() >> 23 & 0xff).cast_signed() - 126;
        if brightest <= 0.0 || exponent + 128 < 1 {
            return 0;
        }
        let scale = 2.0_f32.powi(8 - exponent);
        let mantissa = |value: f32| truncated((value * scale).min(255.0));
        let stored = u32::try_from(exponent + 128).unwrap_or(255).min(255);
        u32::from_le_bytes([
            mantissa(pixel[0]),
            mantissa(pixel[1]),
            mantissa(pixel[2]),
            u8::try_from(stored).unwrap_or(u8::MAX),
        ])
    }

    /// The linear light of each channel of an RGBE word.
    pub(super) fn rgbe_values(word: u32) -> [f64; 3] {
        let [red, green, blue, exponent] = word.to_le_bytes();
        let scale = 2.0_f64.powi(i32::from(exponent) - 136);
        [red, green, blue].map(|mantissa| f64::from(mantissa) * scale)
    }

    /// The whole part of a value from zero to 255.
    fn truncated(value: f32) -> u8 {
        let bits = value.to_bits();
        let exponent = (bits >> 23 & 0xff).cast_signed() - 127;
        if bits >> 31 != 0 || !(0..8).contains(&exponent) {
            return 0;
        }
        let whole = (bits & 0x7f_ffff | 1 << 23) >> (23 - exponent.cast_unsigned());
        u8::try_from(whole).unwrap_or(u8::MAX)
    }
}

/// A picture's samples in the storage of each class.
mod derive {
    use jxr::DecodedSamples;

    use super::{QualityClass, light, split_mix64};

    /// Store the rows of a picture, `width` pixels each, as `class`.
    pub(super) fn samples(class: QualityClass, pixels: &[u8], width: usize) -> DecodedSamples {
        // Only a class made from an RGB picture reads its pixels.
        let rgb = pixels.as_chunks::<3>().0;
        let deep = || deep(pixels);
        let deep_rgb = || deep().as_chunks::<3>().0.to_vec();
        match class {
            QualityClass::Gray8
            | QualityClass::Rgb8
            | QualityClass::Rgb8Chroma422
            | QualityClass::Rgb8Chroma420 => DecodedSamples::U8(pixels.to_vec()),
            QualityClass::Rgba8 => DecodedSamples::U8(with_alpha(rgb).concat()),
            QualityClass::Gray16 | QualityClass::Rgb16 => DecodedSamples::U16(deep()),
            QualityClass::RgbFixed16 => DecodedSamples::I16(
                deep()
                    .into_iter()
                    .map(|sample| i16::try_from(rescaled(sample, 1 << 13)).unwrap_or(i16::MAX))
                    .collect(),
            ),
            QualityClass::RgbFixed32 => DecodedSamples::I32(
                deep()
                    .into_iter()
                    .map(|sample| rescaled(sample, 1 << 24).cast_signed())
                    .collect(),
            ),
            QualityClass::RgbHalf => {
                DecodedSamples::F16(light(pixels).into_iter().map(light::half_bits).collect())
            }
            QualityClass::RgbaHalf => DecodedSamples::F16(
                light(pixels)
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .zip(with_alpha(rgb))
                    .flat_map(|(&[red, green, blue], pixel)| {
                        [red, green, blue, f32::from(pixel[3]) / 255.0]
                    })
                    .map(light::half_bits)
                    .collect(),
            ),
            QualityClass::RgbFloat => DecodedSamples::F32(light(pixels)),
            QualityClass::Rgbe => DecodedSamples::Rgbe(
                light(pixels)
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|&pixel| light::rgbe_word(pixel))
                    .collect(),
            ),
            QualityClass::Rgb101010 => DecodedSamples::Rgb101010(words(&deep_rgb(), [10, 10, 10])),
            QualityClass::Rgb565 => DecodedSamples::Rgb565(short_words(&deep_rgb(), [5, 6, 5])),
            QualityClass::Rgb555 => DecodedSamples::Rgb555(short_words(&deep_rgb(), [5, 5, 5])),
            QualityClass::BlackWhite => DecodedSamples::BitPacked(dithered(pixels, width)),
            QualityClass::Cmyk8 | QualityClass::CmykDirect8 => {
                DecodedSamples::U8(inks(rgb, u8::MAX))
            }
            QualityClass::Cmyk16 => DecodedSamples::U16(inks(&deep_rgb(), u16::MAX)),
            QualityClass::Channels8 => DecodedSamples::U8(
                rgb.iter()
                    .flat_map(|&[red, green, blue]| {
                        [red, green, blue, ycbcr([red, green, blue])[0]]
                    })
                    .collect(),
            ),
            QualityClass::Yuv444 => {
                DecodedSamples::U8(rgb.iter().flat_map(|&pixel| ycbcr(pixel)).collect())
            }
            QualityClass::Yuv422 => DecodedSamples::U8(planar(rgb, width, 1)),
            QualityClass::Yuv420 => DecodedSamples::U8(planar(rgb, width, 2)),
        }
    }

    /// Each sample at 16 bits: 257 times its value, plus detail of up to
    /// half an 8-bit step either way that an 8-bit picture has rounded off.
    ///
    /// The detail is pseudo-random and uniform. That is what the low byte of
    /// a 16-bit photograph mostly holds, and it leaves the picture itself
    /// what the 8-bit classes code.
    fn deep(pixels: &[u8]) -> Vec<u16> {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        pixels
            .iter()
            .map(|&sample| {
                let detail = i32::try_from(split_mix64(&mut state) % 257).unwrap_or(0) - 128;
                let value = (i32::from(sample) * 257 + detail).clamp(0, i32::from(u16::MAX));
                u16::try_from(value).unwrap_or(u16::MAX)
            })
            .collect()
    }

    /// A 16-bit sample on the scale where 65535 becomes `one`, rounded.
    fn rescaled(sample: u16, one: u64) -> u32 {
        let scaled = (u64::from(sample) * one + u64::from(u16::MAX) / 2) / u64::from(u16::MAX);
        u32::try_from(scaled).unwrap_or(u32::MAX)
    }

    /// Linear light of the 16-bit samples read as sRGB.
    fn light(pixels: &[u8]) -> Vec<f32> {
        deep(pixels)
            .into_iter()
            .map(|sample| light::linear(f32::from(sample) / f32::from(u16::MAX)))
            .collect()
    }

    /// Packed words of three fields of the given widths, red uppermost.
    fn words(deep: &[[u16; 3]], bits: [u32; 3]) -> Vec<u32> {
        deep.iter()
            .map(|pixel| {
                pixel.iter().zip(bits).fold(0, |word, (&sample, width)| {
                    word << width | rescaled(sample, (1 << width) - 1)
                })
            })
            .collect()
    }

    fn short_words(deep: &[[u16; 3]], bits: [u32; 3]) -> Vec<u16> {
        words(deep, bits)
            .into_iter()
            .map(|word| u16::try_from(word).unwrap_or(u16::MAX))
            .collect()
    }

    /// Full-range Rec. 601 Y, Cb and Cr of an RGB pixel.
    fn ycbcr(pixel: [u8; 3]) -> [u8; 3] {
        let [red, green, blue] = pixel.map(i32::from);
        let byte = |value: i32| u8::try_from(value.clamp(0, 255)).unwrap_or(u8::MAX);
        [
            byte((77 * red + 150 * green + 29 * blue + 128) >> 8),
            byte(((-43 * red - 85 * green + 128 * blue + 128) >> 8) + 128),
            byte(((128 * red - 107 * green - 21 * blue + 128) >> 8) + 128),
        ]
    }

    /// Y, Cb and Cr planes, chroma averaged over pairs of columns and over
    /// `rows` rows. The dimensions are even along each halved axis.
    fn planar(pixels: &[[u8; 3]], width: usize, rows: usize) -> Vec<u8> {
        let converted: Vec<[u8; 3]> = pixels.iter().map(|&pixel| ycbcr(pixel)).collect();
        let mut planes: Vec<u8> = converted.iter().map(|pixel| pixel[0]).collect();
        for channel in 1..3 {
            for band in converted.chunks_exact(width * rows) {
                for x in (0..width).step_by(2) {
                    let sum: usize = (0..rows)
                        .flat_map(|row| [band[row * width + x], band[row * width + x + 1]])
                        .map(|pixel| usize::from(pixel[channel]))
                        .sum();
                    let count = 2 * rows;
                    planes.push(u8::try_from((sum + count / 2) / count).unwrap_or(u8::MAX));
                }
            }
        }
        planes
    }

    /// Cyan, magenta, yellow and black of RGB samples whose largest value is
    /// `peak`: the complements, with the ink they share moved to black.
    fn inks<T: Copy + Ord + std::ops::Sub<Output = T>>(pixels: &[[T; 3]], peak: T) -> Vec<T> {
        pixels
            .iter()
            .flat_map(|pixel| {
                let complements = pixel.map(|sample| peak - sample);
                let black = complements.into_iter().min().unwrap_or(peak);
                [
                    complements[0] - black,
                    complements[1] - black,
                    complements[2] - black,
                    black,
                ]
            })
            .collect()
    }

    /// RGBA pixels whose alpha is a soft matte of the picture: transparent
    /// where luma is below 64, opaque above 191, and a ramp between.
    fn with_alpha(pixels: &[[u8; 3]]) -> Vec<[u8; 4]> {
        pixels
            .iter()
            .map(|&[red, green, blue]| {
                let ramp = (i32::from(ycbcr([red, green, blue])[0]) - 64) * 2;
                let alpha = u8::try_from(ramp.clamp(0, 255)).unwrap_or(u8::MAX);
                [red, green, blue, alpha]
            })
            .collect()
    }

    /// Rows of one bit per pixel, most significant bit first: white where
    /// the sample exceeds a 4-by-4 ordered-dither threshold.
    fn dithered(pixels: &[u8], width: usize) -> Vec<u8> {
        const BAYER: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
        pixels
            .chunks_exact(width)
            .enumerate()
            .flat_map(|(y, row)| {
                let mut bytes = vec![0_u8; width.div_ceil(8)];
                for (x, &sample) in row.iter().enumerate() {
                    let threshold = 16 * u16::from(BAYER[y % 4][x % 4]) + 8;
                    if u16::from(sample) > threshold {
                        bytes[x / 8] |= 0x80 >> (x % 8);
                    }
                }
                bytes
            })
            .collect()
    }
}

/// The next value of the `SplitMix64` generator whose state is `state`: a
/// fixed, seedable stream of well-mixed 64-bit values for synthetic
/// pictures that every run reproduces.
pub fn split_mix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

#[cfg(test)]
mod tests {
    use jxr::EncodeRequest;

    use super::*;

    /// A small RGB picture with gradients, edges and texture, and its gray
    /// counterpart.
    fn pictures() -> [QualityPicture; 2] {
        let (width, height) = (48_u32, 40_u32);
        let rgb: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let texture = (x * 37 + y * 91) % 23;
                    let edge = if x > 20 && y > 12 { 90 } else { 0 };
                    [x * 5, y * 6 + texture, edge + texture * 3]
                        .map(|value| u8::try_from(value.min(255)).unwrap_or(u8::MAX))
                })
            })
            .collect();
        let gray = rgb.iter().step_by(3).copied().collect();
        let picture = |name: &str, channels, pixels| QualityPicture {
            name: name.to_owned(),
            width,
            height,
            channels,
            pixels,
        };
        [picture("rgb", 3, rgb), picture("gray", 1, gray)]
    }

    #[test]
    fn every_class_stores_one_kind_of_picture_and_reproduces_it_losslessly() {
        let pictures = pictures();
        for class in QualityClass::ALL {
            assert_eq!(QualityClass::from_name(class.name()), Some(class));
            let sources: Vec<_> = pictures
                .iter()
                .filter_map(|picture| class.source(picture))
                .collect();
            let [source] = sources.as_slice() else {
                panic!("{} takes {} pictures", class.name(), sources.len());
            };
            let exact = source.measure(&EncodeRequest::lossless()).unwrap();
            // The default coding of 32-bit samples keeps 14 bits of a
            // fixed-point value up to one and 13 mantissa bits of a float:
            // a small fraction of an 8-bit step.
            if matches!(class, QualityClass::RgbFixed32 | QualityClass::RgbFloat) {
                assert!(exact.max_error < 0.05, "{}", class.name());
            } else {
                assert_eq!(exact.squared_error, 0.0, "{}", class.name());
            }
            let coarse = source.measure(&class.request(40)).unwrap();
            assert!(coarse.bytes <= exact.bytes, "{}", class.name());
        }
    }

    #[test]
    fn float_storage_forms_invert_their_packing() {
        for value in [0.0_f32, 1.0e-6, 0.000_3, 0.02, 0.18, 0.5, 0.999, 1.0] {
            let half = light::half_value(light::half_bits(value));
            assert!((half - value).abs() <= value / 1024.0 + 3.0e-8, "{value}");
            let word = light::rgbe_word([value, value / 3.0, 0.0]);
            let [red, green, blue] = light::rgbe_values(word);
            assert!((red - f64::from(value)).abs() <= f64::from(value) / 128.0);
            assert!((green - f64::from(value) / 3.0).abs() <= f64::from(value) / 128.0);
            assert!(blue.abs() < f64::EPSILON);
        }
        // The sRGB transfer function and its inverse.
        for encoded in [0.0_f32, 0.02, 0.2, 0.5, 1.0] {
            let display = light::display(f64::from(light::linear(encoded)));
            assert!((display - 255.0 * f64::from(encoded)).abs() < 1.0e-3);
        }
    }
}
