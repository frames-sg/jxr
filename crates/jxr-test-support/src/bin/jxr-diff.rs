// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

#[cfg(feature = "metal")]
use jxr_test_support::compare_file_metal;
use jxr_test_support::{T835Oracle, T835ProfileLimit, compare_file, exit_on_error, workspace_root};

fn main() {
    exit_on_error("jxr-diff", run());
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    // Images declaring the Advanced profile need the reference's ceiling
    // raised before it will decode them.
    let advanced = arguments
        .iter()
        .position(|argument| argument == "--advanced")
        .map(|index| arguments.remove(index))
        .is_some();
    let backend = if arguments.first().is_some_and(|value| value == "--backend") {
        if arguments.len() < 2 {
            return Err("--backend requires cpu or metal".into());
        }
        let value = arguments.remove(1);
        arguments.remove(0);
        value
            .to_str()
            .ok_or("backend name must be valid UTF-8")?
            .to_owned()
    } else {
        "cpu".to_owned()
    };
    if backend != "cpu" && backend != "metal" {
        return Err(format!("unknown backend {backend:?}; expected cpu or metal").into());
    }
    #[cfg(not(feature = "metal"))]
    if backend == "metal" {
        return Err("the metal backend requires --features metal".into());
    }
    let inputs = arguments;
    if inputs.is_empty() {
        return Err(
            "usage: jxr-diff [--advanced] [--backend cpu|metal] <image.jxr> [image.jxr ...]".into(),
        );
    }
    let workspace = workspace_root()?;
    let oracle = T835Oracle::for_workspace(workspace).with_profile_limit(if advanced {
        T835ProfileLimit::Advanced
    } else {
        T835ProfileLimit::Main
    });
    #[cfg(feature = "metal")]
    let session = (backend == "metal")
        .then(jxr::metal::MetalDecoderSession::system_default)
        .transpose()?;
    for input in inputs {
        #[cfg(feature = "metal")]
        let compared = if let Some(session) = &session {
            compare_file_metal(&oracle, &input, session)?
        } else {
            compare_file(&oracle, &input)?
        };
        #[cfg(not(feature = "metal"))]
        let compared = compare_file(&oracle, &input)?;
        println!(
            "{}: {backend} {:?}, {}x{}, {} bytes identical",
            input.display(),
            compared.format,
            compared.width,
            compared.height,
            compared.byte_len
        );
    }
    Ok(())
}
