//! CPU feature selection retained by a decoder session.

use fearless_simd::Level;

use crate::executor::Executor;

/// Runtime CPU capabilities detected once when a decoder session is created.
#[derive(Clone, Copy, Debug)]
pub struct CpuCapabilities {
    level: Level,
    executor: Executor,
}

impl CpuCapabilities {
    /// Detect the strongest instruction set supported by the current process,
    /// and decode through the thread pool where the target has threads.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            level: Level::new(),
            executor: Executor::DEFAULT,
        }
    }

    /// The same capabilities with every stage of a decode run on the calling
    /// thread, as it always is on a target without threads (WebAssembly
    /// without the `atomics` feature).
    ///
    /// A decode returns the same result either way.
    #[must_use]
    pub const fn on_calling_thread(self) -> Self {
        Self {
            executor: Executor::Sequential,
            ..self
        }
    }

    pub(crate) const fn level(self) -> Level {
        self.level
    }

    pub(crate) const fn executor(self) -> Executor {
        self.executor
    }
}

/// The vector instruction set the contiguous 32-bit integer kernels
/// (eight-bit plane stores in the decoder, and quantization in the encoder)
/// run on: AVX2 on x86, NEON on `AArch64`, and `simd128` on WebAssembly
/// built with it, where it is a compile-time feature and there is nothing
/// to detect.
///
/// The kernels are written once over [`fearless_simd::Simd`];
/// [`with_i32_vectors`] runs one with the token held here. Each has a
/// checked scalar form, the oracle its tests compare with, which runs
/// wherever there is no vector set.
#[derive(Clone, Copy, Debug)]
pub(crate) enum I32Vectors {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2(fearless_simd::x86::Avx2),
    #[cfg(target_arch = "aarch64")]
    Neon(fearless_simd::aarch64::Neon),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128(fearless_simd::wasm32::WasmSimd128),
    /// No vector set: never made, and present so that the kernels are
    /// compiled and checked on every target.
    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    #[expect(dead_code, reason = "uninhabited: it is never made")]
    None(core::convert::Infallible, fearless_simd::Fallback),
}

impl I32Vectors {
    /// The vector set of `level`, where the kernels have one.
    pub(crate) fn of(level: Level) -> Option<Self> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(avx2) = level.as_avx2() {
            return Some(Self::Avx2(avx2));
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(neon) = level.as_neon() {
            return Some(Self::Neon(neon));
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if let Some(simd128) = level.as_wasm_simd128() {
            return Some(Self::Simd128(simd128));
        }
        // A target without a vector set does not read the level.
        let _ = level;
        None
    }
}

/// Evaluates `$body` with `$simd` bound to the token `$vectors` holds.
macro_rules! with_i32_vectors {
    ($vectors:expr, $simd:ident => $body:expr) => {
        match $vectors {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::cpu::I32Vectors::Avx2($simd) => $body,
            #[cfg(target_arch = "aarch64")]
            $crate::cpu::I32Vectors::Neon($simd) => $body,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            $crate::cpu::I32Vectors::Simd128($simd) => $body,
            #[cfg(not(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "aarch64",
                all(target_arch = "wasm32", target_feature = "simd128")
            )))]
            $crate::cpu::I32Vectors::None(_, $simd) => $body,
        }
    };
}

pub(crate) use with_i32_vectors;

#[cfg(test)]
mod tests {
    use jxr_core::{
        AlphaMode, BitstreamMode, ChannelLayout, DecodeRequest, DecodeScale, DecodeStage,
        EncodeQuantizers, EncodeRequest, OverlapMode, PixelFormat, Rect, SourceImage,
        SourceSamples, StageExecutor,
    };

    use super::{CpuCapabilities, I32Vectors};
    use crate::entropy::encode::test_support::Lcg;
    use crate::{
        CpuDecodeWorkspace, decode_cpu, decode_cpu_with_workspace, encode_annex_a,
        executor::pool_entries, parse_codestream, prepare_plan,
    };

    /// 34 by 33 macroblocks: past every stage's size for sharing out work,
    /// the low-pass plane of a quarter-size decode among them.
    const WIDTH: u32 = 544;
    const HEIGHT: u32 = 528;

    fn pixels() -> Vec<u8> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        (0..WIDTH * HEIGHT * 4)
            .map(|index| {
                state = Lcg::advance(state);
                let ramp = index / 4 % WIDTH / 3 + index / (4 * WIDTH) / 2;
                u8::try_from((ramp + u32::try_from(state >> 60).unwrap()) % 256).unwrap()
            })
            .collect()
    }

    /// What `operation` did that handed tasks to the pool, named.
    fn pooled<R>(name: &str, entered: &mut Vec<String>, operation: impl FnOnce() -> R) -> R {
        let before = pool_entries();
        let result = operation();
        if pool_entries() != before {
            entered.push(name.to_owned());
        }
        result
    }

    /// The decodes of `requests` encoded from one image, on the calling
    /// thread, that handed a task to the pool, by name, each request's
    /// encode among them; frequency-mode images are also decoded at both
    /// reduced scales.
    fn entering_the_pool(requests: &[(&str, EncodeRequest)]) -> Vec<String> {
        let samples = pixels();
        let image = SourceImage::new(
            WIDTH,
            HEIGHT,
            PixelFormat::U8(ChannelLayout::Rgba),
            SourceSamples::U8(&samples),
        )
        .unwrap();
        let cpu = CpuCapabilities::detect().on_calling_thread();
        let rgba = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Rgba));
        let region = Rect {
            x: 37,
            y: 51,
            w: 300,
            h: 211,
        };
        let mut entered = Vec::new();
        for (name, request) in requests {
            let file = pooled(&format!("{name} encode"), &mut entered, || {
                encode_annex_a(&image, request, cpu).unwrap()
            });
            let parsed = parse_codestream(&file).unwrap();
            let mut shapes = vec![
                ("full", rgba.clone()),
                ("region", rgba.clone().with_region(region)),
            ];
            if request.bitstream_mode == BitstreamMode::Frequency {
                shapes.push(("quarter", rgba.clone().with_scale(DecodeScale::Quarter)));
                shapes.push(("sixteenth", rgba.clone().with_scale(DecodeScale::Sixteenth)));
            }
            for (shape, decode) in shapes {
                pooled(&format!("{name} {shape} decode"), &mut entered, || {
                    let plan = prepare_plan(file.len(), &parsed, &decode).unwrap();
                    decode_cpu(&file, &parsed, &plan, &decode, cpu).unwrap()
                });
            }
        }
        entered
    }

    fn frequency(tile_macroblocks: u16) -> EncodeRequest {
        EncodeRequest::lossy(EncodeQuantizers::uniform(10))
            .with_overlap(OverlapMode::Two)
            .with_bitstream_mode(BitstreamMode::Frequency)
            .with_max_tile_macroblocks(tile_macroblocks)
            .with_alpha_mode(AlphaMode::Separate)
    }

    /// A decode and an encode on the calling thread never hand a task to
    /// the pool: every stage of both runs where the session says, the
    /// reduced decode's low-pass overlap filter, the scale probe it runs
    /// through the forward transform and the scaling after it, and the
    /// encoder's reading, analysis and coding among them.
    #[test]
    fn the_calling_thread_runs_every_stage_of_a_decode_and_an_encode() {
        let spatial = EncodeRequest::lossy(EncodeQuantizers::uniform(10))
            .with_overlap(OverlapMode::One)
            .with_max_tile_macroblocks(8)
            .with_alpha_mode(AlphaMode::Integrated);
        let entered = entering_the_pool(&[
            ("one-tile frequency", frequency(u16::MAX)),
            ("tiled spatial", spatial),
        ]);
        assert!(entered.is_empty(), "entered the pool: {entered:?}");
    }

    /// The same of the reading of several tiles by a reduced decode, which
    /// reads them side by side where the session says.
    #[test]
    fn the_calling_thread_reads_the_tiles_of_a_reduced_decode() {
        let entered = entering_the_pool(&[("tiled frequency", frequency(8))]);
        assert!(entered.is_empty(), "entered the pool: {entered:?}");
    }

    fn assert_edge_only_overlap_is_reported_scalar(cpu: CpuCapabilities) {
        let small_samples = vec![0_u8; 16 * 16 * 4];
        let small_image = SourceImage::new(
            16,
            16,
            PixelFormat::U8(ChannelLayout::Rgba),
            SourceSamples::U8(&small_samples),
        )
        .unwrap();
        let small_encoded = encode_annex_a(
            &small_image,
            &EncodeRequest::lossless()
                .with_overlap(OverlapMode::Two)
                .with_alpha_mode(AlphaMode::Integrated),
            cpu,
        )
        .unwrap();
        let small_parsed = parse_codestream(&small_encoded).unwrap();
        let small_decode = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Rgba));
        let small_plan = prepare_plan(small_encoded.len(), &small_parsed, &small_decode).unwrap();
        let small = decode_cpu_with_workspace(
            &small_encoded,
            &small_parsed,
            &small_plan,
            &small_decode,
            cpu,
            &mut CpuDecodeWorkspace::default(),
        )
        .unwrap();
        assert_eq!(
            small
                .report
                .executor_for(DecodeStage::SecondInverseTransform),
            Some(StageExecutor::CpuSimd)
        );
        for stage in [DecodeStage::FirstOverlap, DecodeStage::SecondOverlap] {
            assert_eq!(
                small.report.executor_for(stage),
                Some(StageExecutor::CpuScalar),
                "{stage:?} has only edge or scalar-tail work"
            );
        }
    }

    fn assert_reduced_report_does_not_retain_full_simd(
        cpu: CpuCapabilities,
        image: &SourceImage<'_>,
    ) {
        let encoded = encode_annex_a(
            image,
            &EncodeRequest::lossless()
                .with_bitstream_mode(BitstreamMode::Frequency)
                .with_overlap(OverlapMode::One)
                .with_alpha_mode(AlphaMode::Separate),
            cpu,
        )
        .unwrap();
        let parsed = parse_codestream(&encoded).unwrap();
        let mut workspace = CpuDecodeWorkspace::default();
        let decode = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Rgba));
        let plan = prepare_plan(encoded.len(), &parsed, &decode).unwrap();
        let full =
            decode_cpu_with_workspace(&encoded, &parsed, &plan, &decode, cpu, &mut workspace)
                .unwrap();
        assert_eq!(
            full.report
                .executor_for(DecodeStage::SecondInverseTransform),
            Some(StageExecutor::CpuSimd)
        );
        assert_eq!(
            full.report.executor_for(DecodeStage::SecondOverlap),
            Some(StageExecutor::CpuSimd)
        );

        let reduced = decode.clone().with_scale(DecodeScale::Sixteenth);
        let plan = prepare_plan(encoded.len(), &parsed, &reduced).unwrap();
        let reduced =
            decode_cpu_with_workspace(&encoded, &parsed, &plan, &reduced, cpu, &mut workspace)
                .unwrap();
        for stage in [
            DecodeStage::FirstOverlap,
            DecodeStage::SecondInverseTransform,
            DecodeStage::SecondOverlap,
        ] {
            assert_eq!(
                reduced.report.executor_for(stage),
                Some(StageExecutor::CpuScalar),
                "{stage:?} retained a full-resolution SIMD report"
            );
        }
    }

    fn assert_reduced_overlap_reports_its_vectors(cpu: CpuCapabilities, image: &SourceImage<'_>) {
        let encoded = encode_annex_a(image, &frequency(8), cpu).unwrap();
        let parsed = parse_codestream(&encoded).unwrap();
        let quarter = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Rgba))
            .with_scale(DecodeScale::Quarter);
        let plan = prepare_plan(encoded.len(), &parsed, &quarter).unwrap();
        let quarter = decode_cpu_with_workspace(
            &encoded,
            &parsed,
            &plan,
            &quarter,
            cpu,
            &mut CpuDecodeWorkspace::default(),
        )
        .unwrap();
        assert_eq!(
            quarter.report.executor_for(DecodeStage::FirstOverlap),
            Some(StageExecutor::CpuSimd)
        );
        assert_eq!(
            quarter
                .report
                .executor_for(DecodeStage::SecondInverseTransform),
            Some(StageExecutor::CpuScalar)
        );
    }

    #[test]
    fn reports_only_reconstruction_stages_that_used_native_vectors() {
        let cpu = CpuCapabilities::detect().on_calling_thread();
        if I32Vectors::of(cpu.level()).is_none() {
            return;
        }
        assert_edge_only_overlap_is_reported_scalar(cpu);
        let samples = pixels();
        let image = SourceImage::new(
            WIDTH,
            HEIGHT,
            PixelFormat::U8(ChannelLayout::Rgba),
            SourceSamples::U8(&samples),
        )
        .unwrap();
        assert_reduced_report_does_not_retain_full_simd(cpu, &image);
        assert_reduced_overlap_reports_its_vectors(cpu, &image);
    }
}
