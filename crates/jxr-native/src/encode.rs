//! Source pixels to raw T.832 codestreams.
//!
//! The flow mirrors decoding in reverse: source samples become signed
//! component planes, [`crate::analyze`] transforms and quantizes them, each
//! tile is entropy coded independently, and the headers and tile index are
//! written around the packets. A tile is one packet, or one packet for each
//! band in a frequency-mode codestream. An alpha channel is one more Y-only
//! plane, coded into the same packets or into a codestream of its own.
//!
//! The image passes through those stages as a stream down each component's
//! plane, a few macroblock rows at a time ([`bands`]), so the working memory
//! of an encode grows with the image's width and not its height. The packets are the output: they are
//! appended to the codestream, or to the Annex-A file, as each tile row is
//! finished, and the headers are placed in front of them in the same
//! buffer.

mod bands;
mod input;
mod layout;
mod quality;

use jxr_core::{
    AlphaMode, AnnexAPixelFamily, BandPresence, BitstreamMode, ChromaSampling, ColorFormat,
    EncodeQuantizers, EncodeRequest, EncodedCodestreams, JxrError, JxrErrorKind, OutputBitDepth,
    OverlapMode, QuantizerSet as QuantizerSteps, SourceImage, scaled_after_first_transform,
};
use jxr_math::sampling::DOWNSAMPLED_CENTERING;

use crate::{
    AnnexAWriteOptions, CodestreamHeader, CpuCapabilities, HeaderFlags, ImagePlaneHeader,
    NativeError, QuantizerSet,
    annex_a::{annex_a_alpha_offset, annex_a_primary_offset, complete_annex_a},
    bit_writer::BitWriter,
    entropy::EntropyError,
    header::{inferred_margin, write_codestream_headers},
    metadata::{band_presence_code, internal_color_code, overlap_code},
    pixel_format::output_color_code,
    reconstruct::{ReconstructionConfig, ReconstructionError, TilePartition},
    tail::{PacketBand, write_directory},
    tile_decode::{component_geometry, quantizer::component_steps},
    tile_encode::{PacketOptions, TileEncodeError},
};

use bands::{BandSource, CodestreamPlanes, PlaneAnalysis, code_bands};
use input::{PlaneScaling, PlaneSet, SourceFormat, SourceReader, check_sample_depth};
use layout::{ImageLayout, profile_level};
use quality::ResolvedQuantization;

/// Encode one image as an Annex-A still-image file, the usual `.jxr` form.
///
/// The container declares the pixel format of `image` and holds the
/// codestreams [`encode_codestreams`] returns: one, or two when the alpha
/// channel is coded separately. They are written in place in the file, so
/// no codestream is held beside the file it goes into.
///
/// Every stage runs on the vectors and the executor of `cpu`, as a decode
/// does ([`crate::decode_cpu`]): [`CpuCapabilities::on_calling_thread`]
/// encodes without entering a thread pool. The file is the same either way.
pub fn encode_annex_a(
    image: &SourceImage<'_>,
    request: &EncodeRequest,
    cpu: CpuCapabilities,
) -> Result<Vec<u8>, JxrError> {
    Encoder::new(image, request, cpu)?.annex_a()
}

/// Encode one image as a single raw T.832 codestream.
///
/// The result starts with the `WMPHOTO` signature and can be decoded as is or
/// wrapped in an Annex-A container with [`crate::write_annex_a`]. A source
/// with an alpha channel fits one codestream only as integrated alpha;
/// separate alpha is a second codestream, so such a request is invalid here
/// and [`encode_codestreams`] or [`encode_annex_a`] carries it instead.
/// `cpu` is as for [`encode_annex_a`].
pub fn encode_codestream(
    image: &SourceImage<'_>,
    request: &EncodeRequest,
    cpu: CpuCapabilities,
) -> Result<Vec<u8>, JxrError> {
    let encoder = Encoder::new(image, request, cpu)?;
    if encoder.alpha_mode == AlphaMode::Separate {
        return Err(JxrError::new(
            JxrErrorKind::InvalidRequest,
            "separate alpha does not fit a single codestream",
        ));
    }
    let mut codestream = Vec::new();
    encoder.write_primary(&mut codestream)?;
    Ok(codestream)
}

/// Encode one image as the raw T.832 codestreams an Annex-A container holds.
///
/// A source with an alpha channel yields two codestreams under
/// [`AlphaMode::Separate`], the colour image and a Y-only alpha image, and one
/// codestream holding both planes under [`AlphaMode::Integrated`]. Any other
/// source yields one. `cpu` is as for [`encode_annex_a`].
pub fn encode_codestreams(
    image: &SourceImage<'_>,
    request: &EncodeRequest,
    cpu: CpuCapabilities,
) -> Result<EncodedCodestreams, JxrError> {
    let encoder = Encoder::new(image, request, cpu)?;
    let mut primary = Vec::new();
    encoder.write_primary(&mut primary)?;
    let alpha = if encoder.alpha_mode == AlphaMode::Separate {
        let mut alpha = Vec::new();
        encoder.write_separate_alpha(&mut alpha)?;
        Some(alpha)
    } else {
        None
    };
    Ok(EncodedCodestreams {
        primary,
        alpha,
        pixel_format: encoder.format.annex_a_guid,
    })
}

/// One validated encode: the source, its resolved format, and the geometry
/// every plane and codestream of the image shares.
struct Encoder<'a> {
    image: &'a SourceImage<'a>,
    request: &'a EncodeRequest,
    format: SourceFormat,
    layout: ImageLayout,
    /// How the alpha channel is coded; `None` for a source without one.
    alpha_mode: AlphaMode,
    /// The quantizers and overlap mode the request resolves to for the
    /// source.
    quantization: ResolvedQuantization,
    /// The vectors and executor every stage runs on.
    cpu: CpuCapabilities,
    /// Macroblock rows of each band, and read at a time, when a test sets
    /// them rather than the working bound; they change nothing but the
    /// working memory.
    band_rows: Option<usize>,
}

/// One image plane as its codestream carries it.
struct CodedPlane {
    header: ImagePlaneHeader,
    /// Internal colour format the components are coded in.
    colour: ColorFormat,
    /// How each component is transformed and quantized.
    analysis: PlaneAnalysis,
}

/// What an image header declares about the samples of its codestream.
#[derive(Clone, Copy)]
struct ImageSyntax {
    /// `OUTPUT_CLR_FMT` of the primary plane.
    output_color_code: u8,
    /// Whether packed words are coded red first, as their fields are read.
    packed: bool,
    /// Whether the colour samples are premultiplied by an integrated alpha
    /// plane.
    premultiplied_alpha: bool,
}

/// Longest variable-length word of a tile index entry.
const LONGEST_INDEX_ENTRY: usize = 9;
/// Bytes of the index start code, and of the `SubsequentBytes` word and
/// profile-level record that follow the index.
const DIRECTORY_FRAME_BYTES: usize = 2 + 2 + 4;

impl<'a> Encoder<'a> {
    fn new(
        image: &'a SourceImage<'a>,
        request: &'a EncodeRequest,
        cpu: CpuCapabilities,
    ) -> Result<Self, JxrError> {
        image.validate()?;
        request.validate()?;
        let format = SourceFormat::resolve(
            image.format,
            image.premultiplied_alpha,
            image.sample_bits,
            request,
        )?;
        let alpha_mode = match (format.alpha, request.alpha_mode) {
            (false, _) => AlphaMode::None,
            (true, AlphaMode::None) => {
                return Err(JxrError::new(
                    JxrErrorKind::InvalidRequest,
                    "source alpha channel without an alpha coding mode",
                ));
            }
            (true, mode) => mode,
        };
        let layout = ImageLayout::new(image.width, image.height, request)?;
        let quantization = ResolvedQuantization::resolve(
            request,
            &format,
            second_level_overlap_fits(&format, &layout, request),
        )?;
        let encoder = Self {
            image,
            request,
            format,
            layout,
            alpha_mode,
            quantization,
            cpu,
            band_rows: None,
        };
        // A tile carries one TRIM_FLEXBITS, which the tile decoder here and
        // the T.835 encoder apply to both of its planes. The T.835 and
        // jxrlib decoders read an integrated alpha plane as untrimmed
        // instead, and reject or corrupt such a tile, the T.835 encoder's
        // own included. No coding of the combination is readable everywhere,
        // so none is written; separate alpha carries its own TRIM_FLEXBITS.
        if alpha_mode == AlphaMode::Integrated && encoder.trim_flexbits().is_some() {
            return Err(JxrError::new(
                JxrErrorKind::Unsupported,
                "trimmed flexbits with integrated alpha",
            ));
        }
        encoder.check_subsampled_overlap()?;
        Ok(encoder)
    }

    /// The Annex-A file of the image: the container's directory, then each
    /// codestream written where the file holds it.
    fn annex_a(&self) -> Result<Vec<u8>, JxrError> {
        let separate = self.alpha_mode == AlphaMode::Separate;
        let start = annex_a_primary_offset(separate).map_err(|_| container_error())?;
        let mut file = vec![0; start];
        self.write_primary(&mut file)?;
        let primary = file.len() - start;
        let alpha = if separate {
            let start = annex_a_alpha_offset(primary).map_err(|_| container_error())?;
            file.resize(start, 0);
            self.write_separate_alpha(&mut file)?;
            Some(file.len() - start)
        } else {
            None
        };
        let image = self.image;
        let options = AnnexAWriteOptions::new(image.width, image.height, self.format.annex_a_guid);
        // The container parses what it is given, so a failure here means
        // the encoder produced a codestream its own parser rejects.
        complete_annex_a(&mut file, primary, alpha, &options).map_err(|_| container_error())?;
        Ok(file)
    }

    /// Nothing the T.835 reference software would reject is written: a
    /// request for second-level overlap that the image cannot take is
    /// invalid. See [`second_level_overlap_fits`].
    fn check_subsampled_overlap(&self) -> Result<(), JxrError> {
        if self.quantization.overlap != OverlapMode::Two
            || second_level_overlap_fits(&self.format, &self.layout, self.request)
        {
            return Ok(());
        }
        Err(JxrError::new(
            JxrErrorKind::InvalidRequest,
            "second-level overlap of subsampled chroma across one macroblock",
        ))
    }

    /// The chroma-centering codes the primary plane declares: the grid the
    /// encoder's own decimation produces, or the one a source that is
    /// already subsampled says it has. The vertical code exists for 4:2:0
    /// only.
    fn chroma_centering(&self) -> [u8; 2] {
        let own = DOWNSAMPLED_CENTERING.get();
        match self.format.decimation() {
            Some(ChromaSampling::Cs420) => [own, own],
            Some(_) => [own, 0],
            None => self.image.chroma_centering,
        }
    }

    /// Each plane is scaled by its own losslessness: an alpha plane has its
    /// own header, in its own codestream or beside the primary one.
    fn scaling(&self) -> PlaneScaling {
        PlaneScaling {
            colour: !self.quantization.colour_is_lossless(),
            alpha: !self.quantization.alpha.is_lossless(),
        }
    }

    /// Append the primary codestream to `output`: the colour image, with
    /// the alpha channel as an integrated plane when it is coded so.
    fn write_primary(&self, output: &mut Vec<u8>) -> Result<(), JxrError> {
        // Check all planes once, including any alpha codestream written later.
        check_sample_depth(self.image)?;
        let quantization = &self.quantization;
        let scaled = self.scaling();
        let colour = self.plane(
            self.format.internal_color,
            &|component| quantization.component(component),
            (scaled.colour, self.chroma_centering()),
        )?;
        let integrated = self.alpha_mode == AlphaMode::Integrated;
        let alpha = integrated.then(|| self.alpha_plane()).transpose()?;
        let syntax = ImageSyntax {
            output_color_code: self.format.output_color_code,
            packed: self.format.is_packed(),
            // With separate alpha the container format alone declares it:
            // the primary codestream has no alpha plane to be multiplied by.
            premultiplied_alpha: integrated && self.format.premultiplied_alpha,
        };
        let set = if integrated {
            PlaneSet::All
        } else {
            PlaneSet::Colour
        };
        self.write_codestream(output, syntax, set, &colour, alpha.as_ref())
    }

    /// Append the alpha channel's own codestream to `output`: an ordinary
    /// gray image, read and coded after the colour image is complete.
    fn write_separate_alpha(&self, output: &mut Vec<u8>) -> Result<(), JxrError> {
        let gray = ImageSyntax {
            output_color_code: output_color_code(AnnexAPixelFamily::Luma),
            packed: false,
            premultiplied_alpha: false,
        };
        let alpha = self.alpha_plane()?;
        self.write_codestream(output, gray, PlaneSet::Alpha, &alpha, None)
    }

    /// The alpha channel as a Y-only plane with its own quantizers and
    /// scaled flag.
    fn alpha_plane(&self) -> Result<CodedPlane, JxrError> {
        let quantizers = self.quantization.alpha;
        self.plane(
            ColorFormat::Luma,
            &|_| quantizers,
            (self.scaling().alpha, [0, 0]),
        )
    }

    /// `TRIM_FLEXBITS` of every tile, when the image header enables it.
    fn trim_flexbits(&self) -> Option<u8> {
        (self.request.bands.has_flexbits() && self.request.trim_flexbits != 0)
            .then_some(self.request.trim_flexbits)
    }

    /// The header and analysis of one image plane coded in `colour`, whose
    /// component `component` uses `quantizers(component)`. The image plane
    /// declares the `(scaled flag, chroma centering)` given.
    fn plane(
        &self,
        colour: ColorFormat,
        quantizers: &dyn Fn(usize) -> EncodeQuantizers,
        (scaled, chroma_centering): (bool, [u8; 2]),
    ) -> Result<CodedPlane, JxrError> {
        let (request, layout) = (self.request, &self.layout);
        let components = usize::from(colour.component_count().ok_or_else(|| {
            JxrError::new(JxrErrorKind::InternalInvariant, "encode plane components")
        })?);
        let header = ImagePlaneHeader {
            chroma_centering_x: chroma_centering[0],
            chroma_centering_y: chroma_centering[1],
            ..plane_header(
                colour,
                components,
                quantizers,
                self.format.bit_depth,
                scaled,
                request.bands,
            )?
        };
        let configs = (0..components)
            .map(|component| {
                let geometry = component_geometry(colour, component);
                ReconstructionConfig {
                    macroblock_origin_x: 0,
                    macroblock_origin_y: 0,
                    macroblocks_x: layout.macroblocks_x,
                    macroblocks_y: layout.macroblocks_y,
                    block_columns: geometry.columns,
                    block_rows: geometry.rows,
                    scale_after_first_transform: scaled_after_first_transform(scaled, component),
                    overlap: self.quantization.overlap,
                    tiles: TilePartition {
                        column_boundaries: layout.column_boundaries.clone(),
                        row_boundaries: layout.row_boundaries.clone(),
                        hard_boundaries: request.hard_tiles,
                    },
                }
            })
            .collect();
        let steps = (0..components)
            .map(|component| quantizer_steps(quantizers(component), component, scaled))
            .collect::<Result<_, _>>()?;
        Ok(CodedPlane {
            header,
            colour,
            analysis: PlaneAnalysis {
                configs,
                steps,
                bands: request.bands,
            },
        })
    }

    /// Append a codestream to `output`: the headers of `primary`, with
    /// `alpha` as an integrated alpha plane when given, the tile index, and
    /// the packets of every tile, coded from the planes `set` reads.
    ///
    /// The headers' size is known before any packet is written, the index's
    /// only after the last: room for the longest index is left after the
    /// headers, and closed up once the index is written into it.
    fn write_codestream(
        &self,
        output: &mut Vec<u8>,
        syntax: ImageSyntax,
        set: PlaneSet,
        primary: &CodedPlane,
        alpha: Option<&CodedPlane>,
    ) -> Result<(), JxrError> {
        let layout = &self.layout;
        let header = self.image_header(syntax, alpha.is_some());
        let mut writer = BitWriter::new();
        write_codestream_headers(
            &mut writer,
            &header,
            &primary.header,
            alpha.map(|plane| &plane.header),
        )
        .map_err(header_error)?;
        output.extend_from_slice(&writer.into_bytes());
        let directory_start = output.len();

        let tiles = layout.tile_columns.len() * layout.tile_rows.len();
        let packets =
            tiles * PacketBand::per_tile(header.flags.frequency_mode(), self.request.bands);
        let index = header.flags.index_table_present();
        let reserved = DIRECTORY_FRAME_BYTES + usize::from(index) * LONGEST_INDEX_ENTRY * packets;
        output.resize(directory_start + reserved, 0);

        let source = BandSource {
            reader: SourceReader::new(
                self.image,
                layout,
                self.format,
                self.scaling(),
                set,
                self.cpu.executor(),
            ),
            set,
            decimation: self.format.decimation().filter(|_| set.colour()),
            padded: layout.plane_dimensions()?,
            band_rows: self.band_rows,
        };
        let planes = CodestreamPlanes {
            primary: &primary.analysis,
            alpha: alpha.map(|plane| &plane.analysis),
            options: PacketOptions {
                colour: primary.colour,
                bands: self.request.bands,
                trim_flexbits: self.trim_flexbits(),
                mode: self.request.bitstream_mode,
            },
        };
        let mut lengths = Vec::with_capacity(packets);
        code_bands(&source, &planes, layout, self.cpu, output, &mut lengths)?;

        // The index locates every packet: one for each tile, or one for
        // each band of each tile in a frequency-mode codestream.
        let mut offsets = Vec::with_capacity(lengths.len());
        let mut offset = 0_u64;
        for &length in &lengths {
            offsets.push(offset);
            offset = u64::try_from(length)
                .ok()
                .and_then(|length| offset.checked_add(length))
                .ok_or_else(|| JxrError::arithmetic("encode tile offsets"))?;
        }
        let mut directory = Vec::new();
        write_directory(
            &mut directory,
            index.then_some(&offsets),
            profile_level(&header, &primary.header, layout),
        );
        let unused = reserved.checked_sub(directory.len()).ok_or_else(|| {
            JxrError::new(JxrErrorKind::InternalInvariant, "encode tile index size")
        })?;
        output[directory_start..directory_start + directory.len()].copy_from_slice(&directory);
        output.copy_within(
            directory_start + reserved..,
            directory_start + directory.len(),
        );
        output.truncate(output.len() - unused);
        Ok(())
    }

    fn image_header(&self, syntax: ImageSyntax, alpha_plane: bool) -> CodestreamHeader {
        let (image, layout, request) = (self.image, &self.layout, self.request);
        // The final tile of each axis is implied by the image size.
        let listed = |sizes: &[u16]| sizes[..sizes.len() - 1].to_vec();
        let flags = |index| {
            HeaderFlags::from_parsed([
                request.hard_tiles,
                request.bitstream_mode == BitstreamMode::Frequency,
                index,
                // Coefficients are computed in 32 bits; no 16-bit bound is proven.
                true,
                layout.short_header(image.width, image.height),
                self.trim_flexbits().is_some(),
                syntax.packed,
                syntax.premultiplied_alpha,
                alpha_plane,
            ])
        };
        let mut header = CodestreamHeader {
            width: image.width,
            height: image.height,
            flags: flags(false),
            orientation: 0,
            overlap_mode: overlap_code(self.quantization.overlap),
            output_color_format: syntax.output_color_code,
            output_bit_depth: self.format.bit_depth.header_fields().0,
            tile_widths_mb: listed(&layout.tile_columns),
            tile_heights_mb: listed(&layout.tile_rows),
            margins: [
                0,
                0,
                inferred_margin(image.height),
                inferred_margin(image.width),
            ],
        };
        header.flags = flags(header.requires_index());
        header
    }
}

/// Whether an image can take second-level overlap, which filters subsampled
/// chroma across pairs of macroblocks. The T.835 reference software asserts
/// that no image coded with both is one macroblock wide, and that no tile of
/// one is under hard tiling.
fn second_level_overlap_fits(
    format: &SourceFormat,
    layout: &ImageLayout,
    request: &EncodeRequest,
) -> bool {
    let subsampled = matches!(
        format.internal_color,
        ColorFormat::Yuv(ChromaSampling::Cs420 | ChromaSampling::Cs422)
    );
    let narrow_tile = layout.tile_columns.iter().any(|&width| width < 2);
    !subsampled || !(layout.macroblocks_x < 2 || (request.hard_tiles && narrow_tile))
}

/// Quantization steps the decoder derives from one component's parameters.
///
/// Derived through the decoder's own mapping so the forward quantizer divides
/// by exactly the step reconstruction multiplies by.
fn quantizer_steps(
    parameters: EncodeQuantizers,
    component: usize,
    scaled: bool,
) -> Result<QuantizerSteps, JxrError> {
    component_steps(
        component,
        [parameters.dc, parameters.low_pass, parameters.high_pass],
        scaled,
    )
    .map_err(|_| JxrError::arithmetic("encode quantizer step"))
}

/// The header of a plane of `components` components coded in `colour`.
fn plane_header(
    colour: ColorFormat,
    components: usize,
    quantizers: &dyn Fn(usize) -> EncodeQuantizers,
    bit_depth: OutputBitDepth,
    scaled: bool,
    bands: BandPresence,
) -> Result<ImagePlaneHeader, JxrError> {
    let set = |band: fn(EncodeQuantizers) -> u8| {
        Some(QuantizerSet {
            components: (0..components)
                .map(|component| band(quantizers(component)))
                .collect(),
        })
    };
    let (_, shift_bits, mantissa_length, exponent_bias) = bit_depth.header_fields();
    let internal_color_format = internal_color_code(colour)
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "encode internal colour"))?;
    Ok(ImagePlaneHeader {
        internal_color_format,
        scaled,
        bands_present: band_presence_code(bands),
        components: u16::try_from(components)
            .map_err(|_| JxrError::arithmetic("encode component count"))?,
        chroma_centering_x: 0,
        chroma_centering_y: 0,
        shift_bits,
        mantissa_length,
        exponent_bias,
        dc_quantizers: set(|quantizers| quantizers.dc),
        lp_quantizers: bands
            .has_low_pass()
            .then(|| set(|quantizers| quantizers.low_pass))
            .flatten(),
        hp_quantizers: bands
            .has_high_pass()
            .then(|| set(|quantizers| quantizers.high_pass))
            .flatten(),
    })
}

pub(super) fn reconstruction_error(error: &ReconstructionError) -> JxrError {
    let kind = match error {
        ReconstructionError::ArithmeticOverflow(_) => JxrErrorKind::ArithmeticOverflow,
        ReconstructionError::Unsupported(_) => JxrErrorKind::Unsupported,
        _ => JxrErrorKind::InternalInvariant,
    };
    JxrError::new(kind, "encode forward transform")
}

pub(super) fn tile_error(error: &TileEncodeError) -> JxrError {
    let kind = match error {
        // A coefficient too large for the entropy syntax is an overflow of
        // the representable range, like any other arithmetic limit.
        TileEncodeError::ArithmeticOverflow(_)
        | TileEncodeError::Entropy(EntropyError::CoefficientOverflow) => {
            JxrErrorKind::ArithmeticOverflow
        }
        TileEncodeError::Unsupported(_) => JxrErrorKind::Unsupported,
        TileEncodeError::InvalidInput(_) | TileEncodeError::Entropy(_) => {
            JxrErrorKind::InternalInvariant
        }
    };
    JxrError::new(kind, "encode tile packet")
}

fn header_error(_: NativeError) -> JxrError {
    JxrError::new(JxrErrorKind::InternalInvariant, "encode image header")
}

fn container_error() -> JxrError {
    JxrError::new(JxrErrorKind::InternalInvariant, "encode Annex-A container")
}

#[cfg(test)]
mod tests {
    use jxr_core::{BandPresence, ChannelLayout, PixelFormat, SourceSamples};

    use super::*;
    use crate::entropy::{EntropyError, encode::test_support::Lcg};

    #[test]
    fn encoder_error_mapping_keeps_invariants_distinct_from_coefficient_overflow() {
        let entropy = [
            (
                EntropyError::UnexpectedEnd {
                    bit_position: 0,
                    requested_bits: 1,
                    bit_length: 0,
                },
                JxrErrorKind::InternalInvariant,
            ),
            (
                EntropyError::InvalidBitLength {
                    bit_length: 9,
                    available_bits: 8,
                },
                JxrErrorKind::InternalInvariant,
            ),
            (
                EntropyError::InvalidVlc {
                    syntax: "test",
                    bit_position: 0,
                },
                JxrErrorKind::InternalInvariant,
            ),
            (
                EntropyError::InvalidParameter {
                    parameter: "test",
                    value: 1,
                },
                JxrErrorKind::InternalInvariant,
            ),
            (
                EntropyError::CoefficientOverflow,
                JxrErrorKind::ArithmeticOverflow,
            ),
        ];
        for (error, expected) in entropy {
            assert_eq!(tile_error(&TileEncodeError::Entropy(error)).kind, expected);
        }
        assert_eq!(
            tile_error(&TileEncodeError::ArithmeticOverflow("test")).kind,
            JxrErrorKind::ArithmeticOverflow
        );
        assert_eq!(
            tile_error(&TileEncodeError::Unsupported("test")).kind,
            JxrErrorKind::Unsupported
        );
        assert_eq!(
            tile_error(&TileEncodeError::InvalidInput("test")).kind,
            JxrErrorKind::InternalInvariant
        );
        assert_eq!(
            header_error(NativeError::InvalidSyntax { field: "test" }).kind,
            JxrErrorKind::InternalInvariant
        );
    }

    #[test]
    fn raw_codestream_rejects_separate_alpha_before_scanning_source_values() {
        let samples = [2_000_u16; 4];
        let image = SourceImage::new(
            1,
            1,
            PixelFormat::U16(ChannelLayout::Yuva(ChromaSampling::Cs444)),
            SourceSamples::U16(&samples),
        )
        .unwrap()
        .with_sample_bits(10)
        .unwrap();
        let error = encode_codestream(
            &image,
            &EncodeRequest::lossless(),
            CpuCapabilities::detect().on_calling_thread(),
        )
        .unwrap_err();
        assert_eq!(error.kind, JxrErrorKind::InvalidRequest);
        assert_eq!(
            error.operation,
            "separate alpha does not fit a single codestream"
        );
    }

    /// Samples of an image: smooth ramps under noise, so that every band
    /// of every component codes coefficients.
    fn samples(count: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..count)
            .map(|index| {
                state = Lcg::advance(state);
                let noise = usize::try_from(state >> 59).unwrap();
                u8::try_from((index / 7 % 200 + noise) % 256).unwrap()
            })
            .collect()
    }

    fn lossy(overlap: OverlapMode) -> EncodeRequest {
        EncodeRequest::lossy(EncodeQuantizers {
            dc: 6,
            low_pass: 8,
            high_pass: 12,
        })
        .with_overlap(overlap)
    }

    /// Requests that reach every overlap mode, tiling, packet layout and
    /// chroma sampling, with tiles small enough that bands cross them.
    fn requests() -> Vec<EncodeRequest> {
        let mut requests = vec![
            EncodeRequest::lossless(),
            EncodeRequest::quality(75),
            lossy(OverlapMode::One).with_max_tile_macroblocks(2),
            lossy(OverlapMode::Two),
            lossy(OverlapMode::Two)
                .with_max_tile_macroblocks(2)
                .with_hard_tiles(true),
            lossy(OverlapMode::One)
                .with_max_tile_macroblocks(1)
                .with_hard_tiles(true),
            lossy(OverlapMode::Two)
                .with_max_tile_macroblocks(3)
                .with_bitstream_mode(BitstreamMode::Frequency)
                .with_bands(BandPresence::All)
                .with_trim_flexbits(2),
        ];
        for sampling in [ChromaSampling::Cs422, ChromaSampling::Cs420] {
            requests.push(lossy(OverlapMode::Two).with_chroma_sampling(sampling));
            requests.push(
                lossy(OverlapMode::One)
                    .with_chroma_sampling(sampling)
                    .with_max_tile_macroblocks(2)
                    .with_hard_tiles(true),
            );
        }
        requests
    }

    /// The file `image` encodes to under `request` with `cpu`, in bands of
    /// `rows` macroblock rows.
    fn encode_in_bands(
        image: &SourceImage<'_>,
        request: &EncodeRequest,
        cpu: CpuCapabilities,
        rows: usize,
    ) -> Result<Vec<u8>, JxrError> {
        let mut encoder = Encoder::new(image, request, cpu)?;
        encoder.band_rows = Some(rows);
        encoder.annex_a()
    }

    #[test]
    fn every_band_height_writes_the_file_of_the_whole_image() {
        // Odd sizes, alpha in both modes, planar chroma, and four
        // components, each several macroblock rows tall.
        let formats = [
            (PixelFormat::U8(ChannelLayout::Rgb), 45, 83, false),
            (PixelFormat::U8(ChannelLayout::Bgra), 50, 71, true),
            (PixelFormat::U8(ChannelLayout::Luma), 33, 97, false),
            (PixelFormat::U8(ChannelLayout::Cmyk), 40, 66, false),
            (
                PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420)),
                48,
                80,
                false,
            ),
        ];
        let cpu = CpuCapabilities::detect();
        let mut cases = 0;
        for (seed, (format, width, height, premultiplied)) in (1..).zip(formats) {
            // Planar 4:2:0 is a full plane and two quarter ones.
            let planes = if format.channel_count() == 3
                && matches!(format, PixelFormat::U8(ChannelLayout::Yuv(_)))
            {
                6
            } else {
                4 * usize::from(format.channel_count())
            };
            let count = usize::try_from(width * height).unwrap() * planes / 4;
            let pixels = samples(count, seed);
            let image = SourceImage::new(width, height, format, SourceSamples::U8(&pixels))
                .unwrap()
                .with_premultiplied_alpha(premultiplied)
                .unwrap();
            for request in requests() {
                let alpha_modes: &[AlphaMode] = if format.has_alpha() {
                    &[AlphaMode::Separate, AlphaMode::Integrated]
                } else {
                    &[AlphaMode::Separate]
                };
                for &alpha_mode in alpha_modes {
                    let request = request.clone().with_alpha_mode(alpha_mode);
                    let Ok(whole) = encode_in_bands(&image, &request, cpu, usize::MAX) else {
                        continue;
                    };
                    for rows in [1, 2, 4] {
                        let banded = encode_in_bands(&image, &request, cpu, rows).unwrap();
                        assert!(
                            banded == whole,
                            "{format:?} {width}x{height} {request:?} in bands of {rows}"
                        );
                    }
                    // One thread codes a band a macroblock row at a time.
                    for rows in [2, usize::MAX] {
                        let banded =
                            encode_in_bands(&image, &request, cpu.on_calling_thread(), rows)
                                .unwrap();
                        assert!(
                            banded == whole,
                            "{format:?} {width}x{height} {request:?} in bands of {rows} on \
                             one thread"
                        );
                    }
                    cases += 1;
                }
            }
        }
        // Every request codes every source but a few that the format or
        // geometry refuses.
        assert!(cases >= 50, "{cases} cases coded");
    }
}
