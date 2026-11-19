#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Size gate for the decode-only WebAssembly module.

Builds `crates/jxr-wasm-size`, the decoder as a Flash player links it, for
`wasm32-unknown-unknown` twice: as a browser build without extra target
features runs it, and with the features Ruffle's extension builds enable
(`simd128` among them). It prints each module's size, and fails when the
bytes a browser downloads and compiles, the module without its custom
sections, exceed the budget below.

The budgets sit about one percent above the modules of the change that set
them, so a few hundred bytes of ordinary churn pass and a stage pulled into
the decode path, a generic instantiated once more, or a thread pool linked
back in does not. A change that needs the bytes raises the budget in the
same commit and says why; one that saves bytes may lower it. The sizes are
those of the toolchain `rust-toolchain.toml` pins: a new toolchain moves
them, and its update resets the budgets.

Paths of dependency sources are remapped, so the panic locations in the
module, and with them its size, are the same on every machine.

Usage: python3 tools/wasm_size_check.py
Needs the `wasm32-unknown-unknown` target: rustup target add wasm32-unknown-unknown
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = "wasm32-unknown-unknown"
PACKAGE = "jxr-wasm-size"

# Flavour, the target features it is built with, and its budget in bytes of
# the module without custom sections.
FLAVOURS = (
    ("base", "", 412_000),
    (
        "simd128",
        "-C target-feature=+bulk-memory,+simd128,+nontrapping-fptoint,+sign-ext,+reference-types",
        434_100,
    ),
)

CUSTOM_SECTION = 0
CODE_SECTION = 10
DATA_SECTION = 11


def read_leb(data: bytes, offset: int) -> tuple[int, int]:
    """An unsigned LEB128 value at `offset`, and the offset after it."""
    value, shift = 0, 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def sections(data: bytes) -> list[tuple[int, int, int]]:
    """Each section of a module: its id, and its start and end offsets."""
    if data[:4] != b"\0asm":
        raise ValueError("not a WebAssembly module")
    found, offset = [], 8
    while offset < len(data):
        identifier = data[offset]
        length, body = read_leb(data, offset + 1)
        found.append((identifier, offset, body + length))
        offset = body + length
    if offset != len(data):
        raise ValueError("a section runs past the end of the module")
    return found


def measure(module: Path) -> dict[str, int]:
    """Sizes of a module: whole, without custom sections, code, data, gzip."""
    data = module.read_bytes()
    parts = sections(data)
    kept = data[:8] + b"".join(
        data[start:end] for identifier, start, end in parts if identifier != CUSTOM_SECTION
    )
    section = {identifier: end - start for identifier, start, end in parts}
    sizes = {
        "raw": len(data),
        "loaded": len(kept),
        "code": section.get(CODE_SECTION, 0),
        "data": section.get(DATA_SECTION, 0),
        "gzip": len(gzip.compress(kept, compresslevel=9, mtime=0)),
    }
    brotli = shutil.which("brotli")
    if brotli:
        compressed = subprocess.run(
            [brotli, "--best", "--stdout", "-"], input=kept, capture_output=True, check=True
        )
        sizes["brotli"] = len(compressed.stdout)
    return sizes


def build(flavour: str, features: str) -> Path:
    """Builds the probe for one flavour and returns its module."""
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")).resolve()
    flags = f"{features} --remap-path-prefix={cargo_home}=cargo".strip()
    target_dir = ROOT / "target" / "wasm-size" / flavour
    environment = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    environment["CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS"] = flags
    command = ["cargo", "build", "--release", "--target", TARGET, "-p", PACKAGE]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"error: the {flavour} build of {PACKAGE} failed")
    warnings = [line for line in result.stderr.splitlines() if line.startswith("warning")]
    if warnings:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"error: the {flavour} build of {PACKAGE} has warnings")
    return target_dir / TARGET / "release" / f"{PACKAGE}.wasm"


def main() -> int:
    over = []
    for flavour, features, budget in FLAVOURS:
        sizes = measure(build(flavour, features))
        listed = ", ".join(f"{name} {value:,}" for name, value in sizes.items())
        print(f"{flavour}: {listed}; budget {budget:,} ({budget - sizes['loaded']:+,})")
        if sizes["loaded"] > budget:
            over.append(f"{flavour} module is {sizes['loaded']:,} bytes, over its {budget:,}")
    for message in over:
        print(f"error: {message}; see tools/wasm_size_check.py", file=sys.stderr)
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
