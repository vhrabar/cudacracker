extern crate cc;


use std::{env, path::Path};

fn normalize_arch(value: &str) -> String {
    let cleaned = value.trim();
    let cleaned = cleaned
        .strip_prefix("sm_")
        .or_else(|| cleaned.strip_prefix("compute_"))
        .unwrap_or(cleaned);
    cleaned.to_string()
}

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH_LIST");

    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let archs: Option<Vec<String>> = match env::var("CUDA_ARCH_LIST") {
        Ok(list) if !list.trim().is_empty() => Some(
            list.split(|c| c == ',' || c == ';' || c == ' ' || c == '\t')
                .filter(|s| !s.is_empty())
                .map(normalize_arch)
                .collect(),
        ),
        _ => env::var("CUDA_ARCH")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| vec![normalize_arch(&value)]),
    };

    let mut build = cc::Build::new();
    build.cuda(true).flag("-cudart=shared");

    if let Some(archs) = archs {
        for arch in archs {
            build
                .flag("-gencode")
                .flag(&format!("arch=compute_{arch},code=sm_{arch}"));
        }
    }

    build
        .file("src/gpu_code/md5.cu")
        .compile("libcudacracker.a");

    let lib64 = Path::new(&cuda_home).join("lib64");
    let lib = Path::new(&cuda_home).join("lib");

    if lib64.exists() {
        println!("cargo:rustc-link-search=native={}", lib64.display());
    } else if lib.exists() {
        println!("cargo:rustc-link-search=native={}", lib.display());
    } else {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    }

    println!("cargo:rustc-link-lib=cudart");
}