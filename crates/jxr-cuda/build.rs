// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{env, fs, io, path::PathBuf, process::Command};

fn main() -> io::Result<()> {
    println!("cargo:rustc-check-cfg=cfg(jxr_cuda_oxide)");
    println!("cargo:rerun-if-env-changed=JXR_CUDA_OXIDE_ARCH");
    println!("cargo:rerun-if-env-changed=JXR_CUDA_OXIDE_PTX");
    println!("cargo:rerun-if-env-changed=JXR_REQUIRE_CUDA_OXIDE");
    println!("cargo:rerun-if-changed=src/kernels");
    let math =
        PathBuf::from(env::var_os("DEP_JXR_MATH_MANIFEST_DIR").ok_or_else(|| {
            io::Error::other("Cargo did not provide the jxr-math source directory")
        })?);
    println!("cargo:rerun-if-changed={}", math.display());
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let ptx = out.join("jxr_reconstruction.ptx");
    if let Some(prebuilt) = env::var_os("JXR_CUDA_OXIDE_PTX") {
        println!(
            "cargo:rerun-if-changed={}",
            PathBuf::from(&prebuilt).display()
        );
        fs::copy(prebuilt, &ptx)?;
    } else {
        let available = env::var("HOST").unwrap().contains("linux")
            && Command::new("cargo-oxide")
                .arg("--help")
                .output()
                .is_ok_and(|r| r.status.success());
        if !available {
            if env::var_os("JXR_REQUIRE_CUDA_OXIDE").is_some() {
                return Err(io::Error::other(
                    "CUDA Oxide requires a Linux build host with cargo-oxide installed, or JXR_CUDA_OXIDE_PTX",
                ));
            }
            println!(
                "cargo:warning=CUDA Oxide kernels unavailable; install cargo-oxide on Linux or set JXR_CUDA_OXIDE_PTX to enable CUDA execution"
            );
            fs::write(ptx, "")?;
            return Ok(());
        }
        let project = out.join("cuda-oxide");
        fs::create_dir_all(project.join("simt/src"))?;
        fs::create_dir_all(project.join("src"))?;
        fs::write(project.join("src/main.rs"), "fn main() {}\n")?;
        for name in ["Cargo.toml", "rust-toolchain.toml"] {
            fs::copy(
                manifest.join("src/kernels").join(if name == "Cargo.toml" {
                    "Cargo.toml.in"
                } else {
                    name
                }),
                project.join(name),
            )?;
        }
        let template = fs::read_to_string(manifest.join("src/kernels/simt.toml"))?;

        fs::write(
            project.join("simt/Cargo.toml"),
            template.replace("__JXR_MATH__", math.to_str().unwrap()),
        )?;
        for name in ["main.rs", "reconstruction.rs", "output.rs"] {
            fs::copy(
                manifest.join("src/kernels").join(name),
                project.join("simt/src").join(name),
            )?;
        }
        let arch = env::var("JXR_CUDA_OXIDE_ARCH").unwrap_or_else(|_| "sm_50".into());
        let status = Command::new("cargo")
            .args(["oxide", "build", "--arch", &arch, "--no-fmad"])
            .current_dir(&project)
            .env_remove("RUSTUP_TOOLCHAIN")
            .env_remove("RUSTC")
            .env_remove("RUSTC_WRAPPER")
            .env_remove("RUSTC_WORKSPACE_WRAPPER")
            .env_remove("RUSTDOC")
            .env_remove("RUSTFLAGS")
            .env_remove("CARGO_ENCODED_RUSTFLAGS")
            .env("CARGO_TARGET_DIR", project.join("target"))
            .env("JXR_MATH_DEVICE", "1")
            .status()?;
        if !status.success() {
            return Err(io::Error::other("CUDA Oxide kernel compilation failed"));
        }
        fs::copy(project.join("ptx/jxr_reconstruction.ptx"), &ptx)?;
    }
    let source = fs::read_to_string(ptx)?;
    if !source.contains(".entry jxr_dequantize_first_transform") {
        return Err(io::Error::other(
            "CUDA Oxide artifact omits reconstruction kernels",
        ));
    }
    println!("cargo:rustc-cfg=jxr_cuda_oxide");
    Ok(())
}
