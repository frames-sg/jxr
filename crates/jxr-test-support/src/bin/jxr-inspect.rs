// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{ffi::OsString, path::PathBuf};

use jxr::{BackendRequest, DecodeRequest, DecodeScale};
use jxr_test_support::oracle_format;

#[derive(Debug, PartialEq, Eq)]
struct Options {
    path: PathBuf,
    scale: DecodeScale,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("jxr-inspect: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let Options { path, scale } = parse_options(std::env::args_os().skip(1))?;
    let bytes = std::fs::read(&path)?;
    let parsed = match jxr_native::parse_codestream(&bytes) {
        Ok(parsed) => parsed,
        Err(error) => {
            if let Ok(annex) = jxr_native::parse_annex_a(&bytes) {
                println!("annex-a: {annex:#?}");
                println!(
                    "pixel-format: {:#?}",
                    jxr_native::classify_annex_a_pixel_format(annex.pixel_format_guid)
                );
                if let Ok(headers) =
                    jxr_native::parse_codestream_headers(&bytes[annex.codestream_range.clone()])
                {
                    println!("primary headers: {headers:#?}");
                }
                if let Some(range) = annex.alpha_range
                    && let Ok(headers) = jxr_native::parse_codestream_headers(&bytes[range])
                {
                    println!("separate alpha headers: {headers:#?}");
                }
            }
            return Err(error.into());
        }
    };
    let info = jxr_native::image_info(&parsed)?;
    println!("path: {}", path.display());
    println!("info: {info:#?}");
    println!("directory: {:#?}", parsed.directory);
    let format = oracle_format(&info)?;
    let request = DecodeRequest::new(format.pixel_format)
        .with_alpha(format.alpha)
        .with_scale(scale)
        .with_backend(BackendRequest::Cpu);
    let plan = jxr_native::prepare_plan(bytes.len(), &parsed, &request)?;
    println!(
        "plan: {:?} scale, {:?} source region, {:?} decoded region, {} coefficient bytes",
        plan.scale, plan.output_region, plan.decoded_region, plan.coefficient_bytes
    );
    match jxr_native::tile_decode::decode_tiles(&bytes, &parsed, &plan) {
        Ok(arena) => println!(
            "entropy: {} coefficients, {} macroblocks",
            arena.coefficients.len(),
            arena.macroblocks.len()
        ),
        Err(error) => println!("entropy error: {error:?} ({error})"),
    }
    let decoded = jxr_native::decode_cpu(
        &bytes,
        &parsed,
        &plan,
        &request,
        jxr_native::CpuCapabilities::detect(),
    )?;
    println!(
        "decoded: {:?}, {} bytes, {} plane(s)",
        decoded.decoded_region,
        decoded.samples.byte_len(),
        decoded.planes.len()
    );
    Ok(())
}

fn parse_options(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<Options, Box<dyn std::error::Error>> {
    let mut path = None;
    let mut scale = DecodeScale::Full;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--scale") => {
                scale = match arguments
                    .next()
                    .and_then(|value| value.into_string().ok())
                    .as_deref()
                {
                    Some("full") => DecodeScale::Full,
                    Some("quarter") => DecodeScale::Quarter,
                    Some("sixteenth") => DecodeScale::Sixteenth,
                    _ => return Err("--scale must be `full`, `quarter`, or `sixteenth`".into()),
                };
            }
            Some("--help") => {
                println!("usage: jxr-inspect [--scale full|quarter|sixteenth] IMAGE.jxr");
                std::process::exit(0);
            }
            Some(value) if !value.starts_with('-') && path.is_none() => {
                path = Some(PathBuf::from(argument));
            }
            _ => return Err(format!("unknown argument: {}", argument.to_string_lossy()).into()),
        }
    }
    Ok(Options {
        path: path.ok_or("usage: jxr-inspect [--scale full|quarter|sixteenth] IMAGE.jxr")?,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_option_is_explicit_and_defaults_to_full() {
        assert_eq!(
            parse_options([OsString::from("image.jxr")]).unwrap(),
            Options {
                path: PathBuf::from("image.jxr"),
                scale: DecodeScale::Full,
            }
        );
        assert_eq!(
            parse_options([
                OsString::from("--scale"),
                OsString::from("quarter"),
                OsString::from("image.jxr"),
            ])
            .unwrap()
            .scale,
            DecodeScale::Quarter
        );
        assert!(
            parse_options([
                OsString::from("--scale"),
                OsString::from("eighth"),
                OsString::from("image.jxr"),
            ])
            .is_err()
        );
    }
}
