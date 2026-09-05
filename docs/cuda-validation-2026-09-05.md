# CUDA hardware validation — 2026-09-05

The [CUDA hardware workflow](https://github.com/frames-sg/jxr/actions/runs/33953370733)
passed on commit `233a3265b072ddcab7bf6d6c4626b770cdaf3d74`.
This validates the tested reconstruction backend on one NVIDIA configuration.
It establishes no general speedup over CPU or performance on whole-slide images.

## Environment and scope

- Self-hosted Linux X64 runner `Cuda`, runner group `default`.
- GPU reported as `NVIDIA GeForce RTX 4070 ...`, with 12,282 MiB memory. The
  captured `nvidia-smi` table truncates the model suffix; no more specific model
  is inferred.
- NVIDIA-SMI 610.57.01, KMD 610.88, CUDA UMD 13.3; `nvcc` 13.2.78. The NVRTC
  version and CPU model were not separately captured.
- Default benchmark session: four streams, one warmup, ten measured iterations
  per cell, release build, full readback with CPU checksum verification.
- The first run failed before codec testing because `unzip` was absent. Both
  corpus setup scripts now fall back to Python 3's standard ZIP reader after
  checking the existing pinned SHA-256. No codec or validation policy changed.
  Local tests first reproduced that failure, then passed extraction via both
  tools and rejected corrupted archives for both setup scripts.

## Validation

- Formatting, workspace Clippy, all-feature workspace tests, and the CUDA
  dynamic-loading build passed.
- Three ignored CUDA lifecycle/destination contract tests passed.
- Two ignored hardware tests passed, covering the complete in-scope corpus and
  ROI boundaries against CPU (151.36 seconds for that test executable).
- The T.834/T.835 report has **517 pass, 179 scope exclusions, zero failures**.
  Advanced-only output categories and JPEG 2000 `.jpx` wrappers retain their
  established exclusions; this is not a claim to cover those formats.
- All 20 benchmark cells completed, including CPU checksum checks for every
  warmup and measured readback. Scratch-pool allocation misses were zero after
  warmup in every cell.

## Measurements

The fixture labels are relative: `small` is 32×32 and `large` is only 145×130.
Their centered ROIs are 16×16 and 72×64. They are conformance fixtures, not a
representative pathology corpus. Times include preparation, submission,
synchronization, readback, and checksum verification. Throughput divides batch
size by the median batch time. These are absolute observations, not an
optimization before/after comparison.

| Fixture | Path | Batch | Width | Height | Median batch ms | Images/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| small | full | 1 | 32 | 32 | 0.755 | 1325.2 |
| small | full | 8 | 32 | 32 | 3.950 | 2025.1 |
| small | full | 32 | 32 | 32 | 13.785 | 2321.4 |
| small | full | 64 | 32 | 32 | 26.064 | 2455.5 |
| small | full | 128 | 32 | 32 | 55.893 | 2290.1 |
| small | roi | 1 | 16 | 16 | 0.752 | 1329.9 |
| small | roi | 8 | 16 | 16 | 4.238 | 1887.5 |
| small | roi | 32 | 16 | 16 | 12.821 | 2495.8 |
| small | roi | 64 | 16 | 16 | 25.355 | 2524.2 |
| small | roi | 128 | 16 | 16 | 58.812 | 2176.4 |
| large | full | 1 | 145 | 130 | 2.891 | 345.9 |
| large | full | 8 | 145 | 130 | 7.304 | 1095.3 |
| large | full | 32 | 145 | 130 | 21.822 | 1466.4 |
| large | full | 64 | 145 | 130 | 43.429 | 1473.7 |
| large | full | 128 | 145 | 130 | 80.834 | 1583.5 |
| large | roi | 1 | 72 | 64 | 2.783 | 359.3 |
| large | roi | 8 | 72 | 64 | 6.604 | 1211.4 |
| large | roi | 32 | 72 | 64 | 19.549 | 1636.9 |
| large | roi | 64 | 72 | 64 | 42.784 | 1495.9 |
| large | roi | 128 | 72 | 64 | 79.058 | 1619.1 |

The raw report includes phase medians and p95 values. With only ten iterations,
the tail estimates are too sparse for a reliable latency claim and are omitted
from this summary. GPU background use and filesystem caches were uncontrolled;
2,271 MiB of GPU memory was already occupied in the environment snapshot. Peak
process/device memory was not measured. Transfer counters in the raw report
cover all ten iterations, rather than one image or one batch.

## Reproduction

On the tested pushed branch, dispatch the complete hardware gate:

```sh
gh workflow run cuda-hardware.yml --repo frames-sg/jxr \
  --ref pr/dicom-architecture-2026-09-04
```

Or run the relevant corpus and hardware commands on a compatible NVIDIA host:

```sh
tools/t834-conformance/build.sh
tools/t835-oracle/build.sh
cargo test -p jxr --features cuda --test cuda_contracts -- --ignored --test-threads=1
cargo test -p jxr-test-support --features cuda --test cuda_hardware -- --ignored --test-threads=1
cargo run -p jxr-test-support --features cuda --bin jxr-t834 -- \
  --backend cuda --report target/t834-conformance/reports/ci-cuda.tsv
JXR_BENCH_ITERATIONS=10 cargo run --release -p jxr-test-support \
  --features cuda --bin jxr-cuda-pathology-bench
```

Default sources are `BasicAndOverlap_2x2Tile/Small_Freq_Ov2_2x2_YUV420_QP10.jxr`
and `Windowing/Windowed8.jxr` within the extracted T.834 suite. The workflow's
`jpeg-xr-cuda-hardware-reports` artifact retains the conformance TSV, benchmark
TSV, and GPU/toolkit reports for 14 days. Source archive checksums remain pinned
in the setup scripts.

Raw report SHA-256 values:

- `ci-cuda.tsv`: `b7627055f616d7f81118946f28e46ed6f72bb8ce2230324c8322ebcf468deb63`.
- `cuda-pathology-bench.tsv`: `d534516f720e800c82cbf23217d69dc11d1ee0c53b4bfef8a66d3a59cfc9fe28`.
