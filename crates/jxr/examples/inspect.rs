use std::{env, fs, process::ExitCode};

use jxr::JxrView;

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1) else {
        eprintln!("usage: cargo run -p jxr --example inspect -- <image.jxr>");
        return ExitCode::FAILURE;
    };
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read {}: {error}", path.to_string_lossy());
            return ExitCode::FAILURE;
        }
    };
    match JxrView::parse(&bytes) {
        Ok(view) => {
            println!("{:#?}", view.info());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("failed to inspect {}: {error}", path.to_string_lossy());
            ExitCode::FAILURE
        }
    }
}
