//! The harness run without a fuzzing engine: the cases a format table, a
//! sweep of boundary values, and a pseudo-random sequence give it.

use super::{
    samples::{boundary, top8},
    *,
};

/// Keeps the format table honest: the harness assumes each entry encodes
/// and comes back exactly, in either storage layout and whatever the
/// samples.
#[test]
fn every_declared_format_round_trips_losslessly() {
    let formats = declared_formats();
    assert_eq!(formats.len(), 74);
    for (index, &format) in formats.iter().enumerate() {
        for (width, height) in [(2_u8, 2_u8), (38, 22), (64, 48)] {
            for (loose, pattern) in [(0_u8, 0_u8), (3, 0), (1, 1), (0, 2), (2, 3)] {
                // Format, size, the plain lossless request, row layout,
                // and the sample pattern with its seed.
                let mut bytes = vec![0_u8; 26];
                bytes[0] = u8::try_from(index).unwrap();
                bytes[2..4].copy_from_slice(&[width - 1, height - 1]);
                bytes[9] = loose;
                bytes[12] = pattern;
                bytes[13..26].fill(0x5b ^ width ^ pattern);
                let case = Case::from_bytes(&bytes);
                assert_eq!((case.format, case.sample_bits), format);
                assert_eq!(
                    (case.width, case.height),
                    (u32::from(width), u32::from(height))
                );
                assert_eq!(matches!(case.placement, Placement::Tight), loose == 0);
                assert!(case.must_encode && case.lossless().0, "{case}");
                check(&case);
            }
        }
    }
}

#[test]
fn an_empty_input_is_the_plainest_case() {
    let case = Case::from_bytes(&[]);
    assert_eq!(case.format, PixelFormat::BitPacked(ChannelLayout::Luma));
    assert_eq!((case.width, case.height), (1, 1));
    assert_eq!(case.request, EncodeRequest::lossless());
    assert!(case.must_encode);
    check(&case);
}

thread_local! {
    /// Whether panics on this thread are findings being collected.
    static COLLECTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The message a panic carried.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|&text| text.to_owned()))
        .unwrap_or_else(|| "a panic without a message".to_owned())
}

/// Check every case, each named by what produced it, and fail with the
/// distinct findings: how often each was seen and the first case that
/// showed it. Every case runs whatever the ones before it found.
fn check_all(cases: impl IntoIterator<Item = (String, Case)>) {
    // Findings are counted here, not printed one by one as they panic.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !COLLECTING.get() {
            default_hook(info);
        }
    }));
    COLLECTING.set(true);
    let mut findings = std::collections::BTreeMap::<String, (u64, String)>::new();
    let mut total = 0_u64;
    for (origin, case) in cases {
        total += 1;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(&case)));
        if let Err(panic) = outcome {
            let message = panic_message(panic.as_ref());
            let kind = message.lines().next().unwrap_or_default().to_owned();
            let first = format!("{origin}\n{message}");
            findings.entry(kind).or_insert((0, first)).0 += 1;
        }
    }
    COLLECTING.set(false);
    let report: Vec<String> = findings
        .iter()
        .map(|(kind, (count, first))| format!("{count} x {kind}\n  first: {first}"))
        .collect();
    assert!(
        report.is_empty(),
        "{} of {total} cases failed:\n{}",
        findings.values().map(|finding| finding.0).sum::<u64>(),
        report.join("\n")
    );
}

/// The fuzz target without a fuzzing engine: pseudo-random inputs, half
/// of them stepping through the declared formats so that each is
/// reached however few cases run. `JXR_FUZZ_CASES` sets how many, and
/// `JXR_FUZZ_SEED` where the sequence starts.
#[test]
fn pseudo_random_inputs_hold_every_property() {
    let variable = |name: &str, default: u64| {
        std::env::var(name).map_or(default, |value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be an unsigned integer"))
        })
    };
    let mut noise = Pattern {
        kind: 0,
        state: variable("JXR_FUZZ_SEED", 1),
        step: 0,
        raw: &[],
        position: 0,
    };
    check_all((0..variable("JXR_FUZZ_CASES", 4000)).map(|case| {
        let length = size(noise.noise() % 96);
        let mut bytes: Vec<u8> = (0..length).map(|_| top8(noise.noise())).collect();
        if case % 2 == 0 && length >= 2 {
            // A declared format, and not the rare undeclared choice.
            bytes[0] = (case / 2).to_le_bytes()[0];
            bytes[1] = 0;
        }
        if case % 4 < 2 && length >= 8 {
            // Few options at once: uniform bytes would set half of them.
            bytes[6] &= top8(noise.noise()) & top8(noise.noise());
            bytes[7] &= top8(noise.noise()) & top8(noise.noise());
        }
        let origin = format!("case {case}, input {bytes:02x?}");
        (origin, Case::from_bytes(&bytes))
    }));
}

/// Replays saved fuzz inputs, a corpus or the artifacts of a fuzzing run:
/// every file directly inside the directory `JXR_FUZZ_INPUTS` names. Without
/// the variable there is nothing to replay.
#[test]
fn saved_inputs_hold_every_property() {
    let Some(directory) = std::env::var_os("JXR_FUZZ_INPUTS") else {
        return;
    };
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("JXR_FUZZ_INPUTS cannot be listed: {error}"))
        .map(|entry| entry.expect("a directory entry is readable").path())
        .filter(|path| path.is_file())
        .collect();
    paths.sort();
    check_all(paths.into_iter().map(|path| {
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("{} cannot be read: {error}", path.display()));
        (path.display().to_string(), Case::from_bytes(&bytes))
    }));
}

/// Flat images at every power of two and beside it, in every declared
/// format: the values at which a coefficient or a sample gains a bit,
/// and which noise all but never produces. Each is coded losslessly,
/// with every bit of a deep sample kept, and lossily.
#[test]
fn flat_images_at_every_power_of_two_hold_every_property() {
    let requests = [
        EncodeRequest::lossless(),
        EncodeRequest::lossless().with_shift_bits(0),
        EncodeRequest::quality(50),
    ];
    let mut cases = Vec::new();
    for format in declared_formats() {
        // The smallest image every format allows.
        let dimensions = if format.0.planar_sampling().is_some() {
            (2, 2)
        } else {
            (1, 1)
        };
        for selector in 0..32 * 3 * 2 {
            for (index, request) in requests.iter().enumerate() {
                let recipe = Recipe {
                    format,
                    dimensions,
                    request: request.clone(),
                    representable: true,
                    loose: 0,
                    premultiplied: false,
                    chroma_centering: [0, 0],
                    pattern: Pattern {
                        kind: 5,
                        state: 0,
                        step: selector,
                        raw: &[],
                        position: 0,
                    },
                    must_encode: index == 0,
                };
                let origin = format!("{:#x} everywhere", boundary(selector));
                cases.push((origin, Case::assemble(recipe)));
            }
        }
    }
    check_all(cases);
}
