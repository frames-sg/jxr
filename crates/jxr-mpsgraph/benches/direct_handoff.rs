// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::struct_field_names,
    clippy::too_many_lines,
    reason = "the benchmark keeps the complete equivalent-work matrix explicit"
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use core::{ffi::c_void, ptr::NonNull};
    use std::{path::PathBuf, sync::Arc, time::Instant};

    use jxr::{ChannelLayout, PixelFormat, PreparedJxr};
    use jxr_mpsgraph::{
        MpsGraphBatchDecoder, MpsGraphDecodeInput, MpsGraphDecodeOptions, MpsGraphProgram,
        MpsGraphTensorSpec,
    };
    use objc2::AnyThread;
    use objc2_foundation::{NSArray, NSDictionary, NSNumber};
    use objc2_metal_performance_shaders::MPSDataType;
    use objc2_metal_performance_shaders_graph::MPSGraphTensorData;

    const BATCHES: [usize; 4] = [1, 8, 32, 128];
    const NONBLOCKING_DEPTH: usize = 4;

    #[derive(Clone, Copy)]
    struct Summary {
        median_ms: f64,
        p95_ms: f64,
        ci_low_ms: f64,
        ci_high_ms: f64,
    }

    fn summarize(samples: &mut [f64]) -> Summary {
        samples.sort_by(f64::total_cmp);
        let median_ms = samples[(samples.len() - 1) / 2];
        let p95_ms = samples[(samples.len() * 95).div_ceil(100).saturating_sub(1)];
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = if samples.len() > 1 {
            samples
                .iter()
                .map(|sample| (sample - mean).powi(2))
                .sum::<f64>()
                / (samples.len() - 1) as f64
        } else {
            0.0
        };
        let half_width = 1.96 * (variance / samples.len() as f64).sqrt();
        Summary {
            median_ms,
            p95_ms,
            ci_low_ms: mean - half_width,
            ci_high_ms: mean + half_width,
        }
    }

    fn tensor_bytes(spec: MpsGraphTensorSpec) -> usize {
        spec.shape().into_iter().product::<usize>()
    }

    fn read_u8_tensor(data: &MPSGraphTensorData, spec: MpsGraphTensorSpec) -> Vec<u8> {
        let mut values = vec![0_u8; tensor_bytes(spec)];
        // SAFETY: the codec group is complete, `values` exactly covers the
        // static U8 tensor, and the benchmark intentionally exercises readback.
        unsafe {
            data.mpsndarray().readBytes_strideBytes(
                NonNull::new(values.as_mut_ptr().cast::<c_void>()).expect("nonempty tensor"),
                core::ptr::null_mut(),
            );
        }
        values
    }

    fn read_scores(data: &MPSGraphTensorData, batch: usize) -> f32 {
        let mut values = vec![0.0_f32; batch];
        // SAFETY: graph completion precedes this call and the reference target
        // contains exactly one F32 score per batch image.
        unsafe {
            data.mpsndarray().readBytes_strideBytes(
                NonNull::new(values.as_mut_ptr().cast::<c_void>()).expect("nonempty scores"),
                core::ptr::null_mut(),
            );
        }
        values.into_iter().sum()
    }

    fn staged_iteration(
        decoder: &mut MpsGraphBatchDecoder,
        program: &MpsGraphProgram,
        prepared: &jxr_mpsgraph::MpsGraphPreparedBatch,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let decoded = decoder.decode_prepared(prepared)?;
        if !decoded.group_errors().is_empty() || decoded.groups().len() != 1 {
            return Err(std::io::Error::other("staged decode did not produce one group").into());
        }
        let host = read_u8_tensor(decoded.groups()[0].tensor_data(), program.input_spec());
        let upload = j2k_metal_support::checked_shared_buffer_with_bytes(decoder.device(), &host)?;
        let dimensions = program.input_spec().shape().map(NSNumber::new_usize);
        let shape = NSArray::from_retained_slice(&dimensions);
        // SAFETY: `upload` exactly contains the static U8 tensor and remains
        // retained through the blocking graph invocation below.
        let tensor_data = unsafe {
            MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                MPSGraphTensorData::alloc(),
                &upload,
                &shape,
                MPSDataType::UInt8,
            )
        };
        let feeds = NSDictionary::from_slices(&[program.image_placeholder()], &[&*tensor_data]);
        let targets = NSArray::from_retained_slice(program.targets());
        // SAFETY: all graph objects, feeds, and the upload allocation remain
        // alive until this blocking call returns its target dictionary.
        let results = unsafe {
            program
                .graph()
                .runWithMTLCommandQueue_feeds_targetTensors_targetOperations(
                    decoder.command_queue(),
                    &feeds,
                    &targets,
                    None,
                )
        };
        let result = results
            .objectForKey(program.targets()[0].as_ref())
            .ok_or_else(|| std::io::Error::other("staged graph omitted its target"))?;
        Ok(read_scores(&result, program.input_spec().shape()[0]))
    }

    let mut arguments = std::env::args_os().skip(1).map(PathBuf::from);
    let paths = [256_u32, 512, 1024].map(|size| {
        arguments.next().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("usage: direct_handoff <{size}px.jxr> <512px.jxr> <1024px.jxr>"),
            )
        })
    });
    let paths = paths.into_iter().collect::<Result<Vec<_>, _>>()?;
    if arguments.next().is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "direct_handoff accepts exactly three input paths",
        )
        .into());
    }
    let iterations = std::env::var("JXR_MPSGRAPH_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(if cfg!(debug_assertions) { 1 } else { 10 })
        .max(1);

    println!("size,batch,path,cold_ms,median_ms,p95_ms,ci95_low_ms,ci95_high_ms,checksum");
    for (&size, path) in [256_u32, 512, 1024].iter().zip(paths) {
        let bytes = Arc::<[u8]>::from(std::fs::read(&path)?);
        let image = PreparedJxr::from_arc(bytes)?;
        if image.info().dimensions() != (size, size) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} is not {size}x{size}", path.display()),
            )
            .into());
        }
        for batch in BATCHES {
            let mut decoder = MpsGraphBatchDecoder::system_default()?;
            let inputs = (0..batch)
                .map(|_| MpsGraphDecodeInput {
                    image: image.clone(),
                    options: MpsGraphDecodeOptions::new(PixelFormat::U8(ChannelLayout::Rgb)),
                })
                .collect();
            let prepared = decoder.prepare(inputs)?;
            if !prepared.errors().is_empty() || prepared.groups().len() != 1 {
                return Err(std::io::Error::other(
                    "benchmark input did not prepare as one RGB8 group",
                )
                .into());
            }
            let group = &prepared.groups()[0];
            let program =
                MpsGraphProgram::rgb8_nhwc_reference(batch, size as usize, size as usize)?;
            if program.input_spec() != group.spec() {
                return Err(std::io::Error::other("benchmark graph/group mismatch").into());
            }

            let cold_started = Instant::now();
            let cold_output = decoder.run_prepared_group(&program, group)?;
            let cold_ms = cold_started.elapsed().as_secs_f64() * 1_000.0;
            let expected = read_scores(&cold_output.results()[0], batch);
            let mut summaries = Vec::new();

            for path_name in ["staged", "completed", "pipelined", "nonblocking"] {
                let mut samples = Vec::with_capacity(iterations);
                let mut checksum = 0.0_f64;
                for _ in 0..iterations {
                    let started = Instant::now();
                    let score = match path_name {
                        "staged" => staged_iteration(&mut decoder, &program, &prepared)?,
                        "completed" => {
                            let decoded = decoder.decode_prepared(&prepared)?;
                            let (mut groups, errors, group_errors) = decoded.into_parts();
                            if !errors.is_empty() || !group_errors.is_empty() || groups.len() != 1 {
                                return Err(std::io::Error::other(
                                    "completed path did not produce one group",
                                )
                                .into());
                            }
                            let output = program
                                .submit_completed(decoder.command_queue(), groups.remove(0))?
                                .wait()?;
                            read_scores(&output.results()[0], batch)
                        }
                        "pipelined" => {
                            let output = decoder.run_prepared_group(&program, group)?;
                            read_scores(&output.results()[0], batch)
                        }
                        "nonblocking" => {
                            let submitted = (0..NONBLOCKING_DEPTH)
                                .map(|_| decoder.submit_prepared_group(&program, group))
                                .collect::<Result<Vec<_>, _>>()?;
                            let sum = submitted.into_iter().try_fold(0.0_f32, |sum, run| {
                                let output = run.wait()?;
                                Ok::<_, jxr_mpsgraph::Error>(
                                    sum + read_scores(&output.results()[0], batch),
                                )
                            })?;
                            sum / NONBLOCKING_DEPTH as f32
                        }
                        _ => unreachable!(),
                    };
                    if (score - expected).abs() > 1.0e-4 * expected.abs().max(1.0) {
                        return Err(std::io::Error::other(
                            "benchmark paths produced different graph results",
                        )
                        .into());
                    }
                    checksum += f64::from(std::hint::black_box(score));
                    let divisor = if path_name == "nonblocking" {
                        NONBLOCKING_DEPTH as f64
                    } else {
                        1.0
                    };
                    samples.push(started.elapsed().as_secs_f64() * 1_000.0 / divisor);
                }
                let summary = summarize(&mut samples);
                println!(
                    "{size},{batch},{path_name},{cold_ms:.3},{:.3},{:.3},{:.3},{:.3},{checksum:.6}",
                    summary.median_ms, summary.p95_ms, summary.ci_low_ms, summary.ci_high_ms,
                );
                summaries.push((path_name, summary));
            }
            let staged = summaries[0].1;
            let pipelined = summaries[2].1;
            let qualifies = iterations >= 2
                && pipelined.median_ms <= staged.median_ms * 0.90
                && pipelined.ci_high_ms < staged.ci_low_ms;
            println!("{size},{batch},speed_claim_qualified,{cold_ms:.3},0,0,0,0,{qualifies}");
        }
    }
    Ok(())
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn main() {
    println!("size,batch,path,cold_ms,median_ms,p95_ms,ci95_low_ms,ci95_high_ms,checksum");
    for size in [256, 512, 1024] {
        for batch in [1, 8, 32, 128] {
            for path in ["staged", "completed", "pipelined", "nonblocking"] {
                println!(
                    "{size},{batch},{path},unsupported,unsupported,unsupported,unsupported,unsupported,0"
                );
            }
        }
    }
}
