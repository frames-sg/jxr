use std::{env, fmt::Write, fs, io, path::PathBuf};

const INPUT: &str = "data/reconstruction.abi";

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed={INPUT}");
    let source = fs::read_to_string(INPUT)?;
    let permutation = parse_permutation(&source)?;
    let abi_structs = parse_abi_structs(&source)?;
    let generated = generate(&permutation, &abi_structs)
        .map_err(|_| io::Error::other("failed to render reconstruction bindings"))?;
    let output = PathBuf::from(required_env("OUT_DIR")?).join("reconstruction_tables.rs");
    fs::write(output, generated)
}

fn generate(
    permutation: &[usize; 16],
    abi_structs: &[AbiStruct],
) -> Result<String, std::fmt::Error> {
    let values = permutation
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut generated = String::new();
    writeln!(
        generated,
        "pub const INVERSE_PERMUTATION: [usize; 16] = [{values}];"
    )?;
    for abi in abi_structs {
        let constant = constant_name(&abi.name);
        writeln!(
            generated,
            "pub const ABI_{constant}_SIZE: usize = {};",
            abi.fields.len() * 4
        )?;
        for (index, field) in abi.fields.iter().enumerate() {
            writeln!(
                generated,
                "pub const ABI_{constant}_{}_OFFSET: usize = {};",
                constant_name(field),
                index * 4
            )?;
        }
    }
    writeln!(
        generated,
        "pub const METAL_RECONSTRUCTION_CONSTANTS: &str = r\""
    )?;
    writeln!(
        generated,
        "constant uint JXR_INVERSE_PERMUTATION[16] = {{{values}}};"
    )?;
    for abi in abi_structs {
        writeln!(generated, "struct {} {{", abi.name)?;
        for field in &abi.fields {
            writeln!(generated, "    uint {field};")?;
        }
        writeln!(generated, "}};")?;
    }
    writeln!(generated, "\";")?;
    writeln!(
        generated,
        "pub const CUDA_RECONSTRUCTION_CONSTANTS: &str = r\""
    )?;
    writeln!(
        generated,
        "__device__ __constant__ unsigned int JXR_INVERSE_PERMUTATION[16] = {{{values}}};"
    )?;
    for abi in abi_structs {
        writeln!(generated, "struct {} {{", abi.name)?;
        for field in &abi.fields {
            writeln!(generated, "    unsigned int {field};")?;
        }
        writeln!(generated, "}};")?;
    }
    writeln!(generated, "\";")?;
    Ok(generated)
}

struct AbiStruct {
    name: String,
    fields: Vec<String>,
}

fn parse_abi_structs(source: &str) -> io::Result<Vec<AbiStruct>> {
    source
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("abi."))
        .map(|line| {
            let (name, fields) = line
                .split_once('=')
                .ok_or_else(|| io::Error::other("invalid abi declaration"))?;
            let name = name
                .trim()
                .strip_prefix("abi.")
                .filter(|name| !name.is_empty())
                .ok_or_else(|| io::Error::other("invalid abi name"))?;
            let fields = fields
                .split(',')
                .map(|field| {
                    let (name, ty) = field
                        .trim()
                        .split_once(':')
                        .ok_or_else(|| io::Error::other("invalid abi field"))?;
                    if ty != "u32" || name.is_empty() {
                        return Err(io::Error::other("only named u32 ABI fields are supported"));
                    }
                    Ok(name.to_owned())
                })
                .collect::<io::Result<Vec<_>>>()?;
            if fields.is_empty() {
                return Err(io::Error::other("abi struct has no fields"));
            }
            Ok(AbiStruct {
                name: name.to_owned(),
                fields,
            })
        })
        .collect()
}

fn constant_name(name: &str) -> String {
    name.bytes()
        .map(|byte| char::from(byte.to_ascii_uppercase()))
        .collect()
}

fn parse_permutation(source: &str) -> io::Result<[usize; 16]> {
    let value = source
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("inverse_permutation ="))
        .ok_or_else(|| io::Error::other("reconstruction ABI omits inverse_permutation"))?;
    let parsed = value
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<usize>()
                .map_err(|_| io::Error::other("inverse_permutation contains a non-integer"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let permutation: [usize; 16] = parsed
        .try_into()
        .map_err(|_| io::Error::other("inverse_permutation must contain 16 values"))?;
    let mut seen = [false; 16];
    for &index in &permutation {
        let slot = seen
            .get_mut(index)
            .ok_or_else(|| io::Error::other("inverse_permutation value exceeds 15"))?;
        if *slot {
            return Err(io::Error::other("inverse_permutation contains a duplicate"));
        }
        *slot = true;
    }
    Ok(permutation)
}

fn required_env(name: &str) -> io::Result<std::ffi::OsString> {
    env::var_os(name).ok_or_else(|| io::Error::other(format!("Cargo omitted {name}")))
}
