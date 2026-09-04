# Third-party notices

Normative VLC, scan, interpolation, and transform-permutation values are
transcribed from ITU-T T.832. No T.835/JXRLib source code is included.

JPEG XR tables added from the ITU-T T.835 / Microsoft JPEG XR reference source
must retain the applicable BSD-3-Clause copyright and disclaimer here.

`fearless_simd` 0.7.0 is used through its safe capability-token API and is
available under MIT or Apache-2.0 licensing.

`cudarc` 0.19.9 provides dynamically loaded CUDA Driver API and NVRTC bindings
for the optional `jxr-cuda` crate and is available under MIT or Apache-2.0
licensing. No NVIDIA toolkit code or headers are distributed by this repository.
Its dynamic loader dependency, `libloading` 0.9.0, is available under the ISC
license; `cfg-if` and the Windows loader shim are MIT OR Apache-2.0.
