//! Encode a binary PGM or PPM image as a JPEG XR file.
//!
//! ```sh
//! cargo run -p jxr --example encode -- input.ppm output.jxr [quality]
//! cargo run -p jxr --example encode -- input.ppm output.jxr --quantizer N
//! ```
//!
//! Without a third argument the image is coded losslessly. A quality from 0
//! to 100 selects a point on the encoder's quality scale: 100 is lossless and
//! smaller values give smaller files. `--quantizer N` instead applies the
//! T.832 quantization parameter `N`, from 2 to 255, to every band; larger
//! values give smaller files.

use jxr::{
    ChannelLayout, EncodeQuantizers, EncodeRequest, PixelFormat, SourceImage, SourceSamples,
};

const USAGE: &str =
    "usage: encode <input.pgm|input.ppm> <output.jxr> [quality 0-100 | --quantizer 2-255]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (input, output, request) = match arguments.as_slice() {
        [input, output] => (input, output, EncodeRequest::lossless()),
        [input, output, quality] => {
            let quality = quality.parse::<u8>().ok().filter(|quality| *quality <= 100);
            let quality = quality.ok_or(USAGE)?;
            (input, output, EncodeRequest::quality(quality))
        }
        [input, output, flag, parameter] if flag == "--quantizer" => {
            let parameter = parameter.parse::<u8>().map_err(|_| USAGE)?;
            let quantizers = EncodeQuantizers::uniform(parameter);
            (input, output, EncodeRequest::lossy(quantizers))
        }
        _ => return Err(USAGE.into()),
    };
    let bytes = std::fs::read(input)?;
    let (width, height, layout, samples) = parse_pnm(&bytes)?;
    let image = SourceImage::new(
        width,
        height,
        PixelFormat::U8(layout),
        SourceSamples::U8(samples),
    )?;
    let encoded = jxr::encode(&image, &request)?;
    std::fs::write(output, &encoded)?;
    println!(
        "{width}x{height} {layout:?}: {} bytes to {} bytes",
        samples.len(),
        encoded.len()
    );
    Ok(())
}

/// Width, height, channel layout, and samples of a parsed PNM image.
type Pnm<'a> = (u32, u32, ChannelLayout, &'a [u8]);

/// Parse an 8-bit binary PGM (`P5`) or PPM (`P6`) into its samples.
fn parse_pnm(bytes: &[u8]) -> Result<Pnm<'_>, Box<dyn std::error::Error>> {
    let mut position = 0;
    let mut field = || -> Result<&[u8], Box<dyn std::error::Error>> {
        loop {
            while bytes.get(position).is_some_and(u8::is_ascii_whitespace) {
                position += 1;
            }
            if bytes.get(position) != Some(&b'#') {
                break;
            }
            while bytes.get(position).is_some_and(|&byte| byte != b'\n') {
                position += 1;
            }
        }
        let start = position;
        while bytes
            .get(position)
            .is_some_and(|byte| !byte.is_ascii_whitespace())
        {
            position += 1;
        }
        if start == position {
            return Err("truncated PNM header".into());
        }
        Ok(&bytes[start..position])
    };
    let layout = match field()? {
        b"P5" => ChannelLayout::Luma,
        b"P6" => ChannelLayout::Rgb,
        _ => return Err("only binary PGM (P5) and PPM (P6) are supported".into()),
    };
    let number = |field: &[u8]| -> Result<u32, Box<dyn std::error::Error>> {
        Ok(std::str::from_utf8(field)?.parse()?)
    };
    let width = number(field()?)?;
    let height = number(field()?)?;
    if number(field()?)? != 255 {
        return Err("only 8-bit PNM samples are supported".into());
    }
    // Exactly one whitespace byte separates the header from the samples.
    let samples = bytes.get(position + 1..).ok_or("truncated PNM samples")?;
    Ok((width, height, layout, samples))
}
