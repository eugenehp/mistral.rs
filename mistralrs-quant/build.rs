#[cfg(feature = "cuda")]
#[allow(unused)]
fn cuda_version_from_build_system() -> (usize, usize) {
    let output = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .expect("Failed to execute `nvcc`");

    if !output.status.success() {
        panic!(
            "`nvcc --version` failed.\nstdout:\n{}\n\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_line = stdout.lines().nth(3).unwrap();
    let release_section = version_line.split(", ").nth(1).unwrap();
    let version_number = release_section.split(' ').nth(1).unwrap();

    match version_number {
        "13.1" => (13, 1),
        "13.0" => (13, 0),
        "12.9" => (12, 9),
        "12.8" => (12, 8),
        "12.6" => (12, 6),
        "12.5" => (12, 5),
        "12.4" => (12, 4),
        "12.3" => (12, 3),
        "12.2" => (12, 2),
        "12.1" => (12, 1),
        "12.0" => (12, 0),
        "11.8" => (11, 8),
        "11.7" => (11, 7),
        "11.6" => (11, 6),
        "11.5" => (11, 5),
        "11.4" => (11, 4),
        v => panic!("Unsupported cuda toolkit version: `{v}`. Please raise a github issue."),
    }
}

fn main() -> Result<(), String> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_marlin_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_blockwise_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_scalar_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_vector_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_wmma_kernels)");

    #[cfg(feature = "cuda")]
    {
        use std::{path::PathBuf, vec};
        const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

        println!("cargo:rerun-if-changed=build.rs");
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        let mut builder = cudaforge::KernelBuilder::new()
            .source_glob("kernels/*/*.cu")
            .out_dir(build_dir.clone())
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--verbose")
            .arg("--compiler-options")
            .arg("-fPIC");

        let compute_cap = builder.get_compute_cap().unwrap_or(80);
        // ======== Handle optional kernel compilation via rustc-cfg flags
        let cc_over_80 = compute_cap >= 80;

        if cc_over_80 {
            println!("cargo:rustc-cfg=has_marlin_kernels");
            println!("cargo:rustc-cfg=has_blockwise_fp8_kernels");
            println!("cargo:rustc-cfg=has_scalar_fp8_kernels");
            println!("cargo:rustc-cfg=has_vector_fp8_kernels");
            // WMMA tensor core MXFP4 kernel (FP16/BF16 WMMA requires SM >= 80)
            println!("cargo:rustc-cfg=has_mxfp4_wmma_kernels");
        }
        // MXFP4 is always enabled with CUDA (uses LUT-based dequantization)
        println!("cargo:rustc-cfg=has_mxfp4_kernels");

        let excluded_files = if cc_over_80 {
            vec!["dummy_*.cu", "*_dummy.cu"]
        } else {
            vec!["marlin_*.cu", "*_fp8.cu", "*_fp8_gemm.cu", "*_wmma.cu"]
        };
        builder = builder.exclude(&excluded_files);

        // https://github.com/EricLBuehler/mistral.rs/issues/286
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            builder = builder.arg("--compiler-options");
            builder = builder.arg(cuda_nvcc_flags_env);
        }

        let target = std::env::var("TARGET").unwrap();
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        // https://github.com/EricLBuehler/mistral.rs/issues/588
        let out_file = if target.contains("msvc") {
            // Windows case
            build_dir.join("mistralrsquant.lib")
        } else {
            build_dir.join("libmistralrsquant.a")
        };
        builder
            .build_lib(out_file)
            .expect("Build mistral quant lib failed!");
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=mistralrsquant");
        println!("cargo:rustc-link-lib=dylib=cudart");

        if target.contains("msvc") {
            // nothing to link to
        } else if target.contains("apple")
            || target.contains("freebsd")
            || target.contains("openbsd")
        {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else if target.contains("android") {
            println!("cargo:rustc-link-lib=dylib=c++_shared");
        } else {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }

        let (major, minor) = cuda_version_from_build_system();
        println!("cargo:rustc-cfg=feature=\"cuda-{major}0{minor}0\"");

        Ok(())
    }

    #[cfg(feature = "metal")]
    {
        use std::path::PathBuf;
        use std::process::Command;
        use std::{env, str};

        const METAL_SOURCES: [&str; 15] = [
            "bitwise",
            "blockwise_fp8",
            "bnb_dequantize",
            "f8q8",
            "fused_glu",
            "hqq_dequantize",
            "hqq_bitpack",
            "mxfp4",
            "quantized",
            "scalar_fp8",
            "scan",
            "sdpa_with_sinks",
            "softmax_with_sinks",
            "sort",
            "copy",
        ];
        const HEADER_SOURCES: [&str; 5] = ["utils", "bf16", "scan_impl", "sort_impl", "copy_impl"];
        // Include-only headers (not compiled directly, just tracked for changes)
        const INCLUDE_ONLY: [&str; 2] = ["float8", "float4"];
        for src in METAL_SOURCES {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        for src in HEADER_SOURCES {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        for src in INCLUDE_ONLY {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        println!("cargo::rerun-if-changed=build.rs");

        // Check if precompilation should be skipped
        // https://github.com/EricLBuehler/mistral.rs/pull/1311#issuecomment-3001309885
        println!("cargo:rerun-if-env-changed=MISTRALRS_METAL_PRECOMPILE");
        let skip_precompile = env::var("MISTRALRS_METAL_PRECOMPILE")
            .map(|v| v == "0" || v.to_lowercase() == "false")
            .unwrap_or(false);

        if skip_precompile {
            println!(
                "cargo:warning=Skipping Metal kernel precompilation (MISTRALRS_METAL_PRECOMPILE=0)"
            );
            // Write a dummy metallib file to satisfy the include_bytes! macro
            let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
            std::fs::write(out_dir.join("mistralrs_quant.metallib"), []).unwrap();
            std::fs::write(out_dir.join("mistralrs_quant_ios.metallib"), []).unwrap();
            std::fs::write(out_dir.join("mistralrs_quant_tvos.metallib"), []).unwrap();
            return Ok(());
        }

        enum Platform {
            MacOS,
            Ios,
            TvOS,
        }

        impl Platform {
            fn sdk(&self) -> &str {
                match self {
                    Platform::MacOS => "macosx",
                    Platform::Ios => "iphoneos",
                    Platform::TvOS => "appletvos",
                }
            }

            fn metal_std(&self) -> &str {
                // Use Metal 3.1 unified standard for all platforms.
                // This fixes Xcode 26+ where the default Metal standard may be too low.
                // https://github.com/EricLBuehler/mistral.rs/issues/1844
                //
                // Note: tvOS devices with A15+ (Apple TV 4K 3rd gen) support Metal 3.1.
                match self {
                    Platform::MacOS | Platform::Ios | Platform::TvOS => "metal3.1",
                }
            }
        }

        fn compile(platform: Platform) -> Result<(), String> {
            let current_dir = env::current_dir().expect("Failed to get current directory");
            let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
            let working_directory = out_dir.to_string_lossy().to_string();
            let sources = current_dir.join("src").join("metal_kernels");

            // Compile metal to air
            let mut compile_air_cmd = Command::new("xcrun");
            compile_air_cmd
                .arg("--sdk")
                .arg(platform.sdk())
                .arg("metal")
                .arg(format!("-std={}", platform.metal_std()))
                .arg(format!("-working-directory={working_directory}"))
                .arg("-Wall")
                .arg("-Wextra")
                .arg("-O3")
                .arg("-c")
                .arg("-w");
            for metal_file in METAL_SOURCES {
                compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
            }
            for metal_file in HEADER_SOURCES {
                compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
            }
            // Spawn the Metal→AIR compilation once and wait for it to finish,
            // checking the exit code.  (Previously this was spawned twice — the
            // first call discarded the exit status and the second call repeated
            // the work — which could produce stale .air files on a failed build.)
            let status = compile_air_cmd
                .spawn()
                .expect("Failed to spawn Metal→AIR compiler (xcrun metal -c)")
                .wait()
                .expect("Failed to wait for Metal→AIR compiler");
            if !status.success() {
                panic!("Compiling metal → air failed. Exit status: {status}");
            }

            // ── AIR → metallib ────────────────────────────────────────────────
            //
            // Use `xcrun --sdk <sdk> metallib` (the dedicated metallib linker),
            // NOT `xcrun metal`, for the linking step.  On Xcode 15/16+ the
            // `metal` driver used without `--sdk` can produce a valid-but-empty
            // 118-byte metallib stub and still exit 0, which silently breaks
            // everything downstream (embedded via include_bytes!, loaded without
            // error, but no kernel functions found at runtime).
            let lib_name = match platform {
                Platform::MacOS => "mistralrs_quant.metallib",
                Platform::Ios => "mistralrs_quant_ios.metallib",
                Platform::TvOS => "mistralrs_quant_tvos.metallib",
            };
            let metallib = out_dir.join(lib_name);
            let mut compile_metallib_cmd = Command::new("xcrun");
            compile_metallib_cmd
                .arg("--sdk")
                .arg(platform.sdk())
                .arg("metallib")
                .arg("-o")
                .arg(&metallib);

            for metal_file in METAL_SOURCES {
                compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
            }
            // Note: header-only .air files (utils, bf16, scan_impl, …) are
            // included transitively; linking them separately is harmless.
            for metal_file in HEADER_SOURCES {
                compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
            }

            let status = compile_metallib_cmd
                .spawn()
                .expect("Failed to spawn AIR→metallib linker (xcrun metallib)")
                .wait()
                .expect("Failed to wait for AIR→metallib linker");
            if !status.success() {
                panic!("Compiling air → metallib failed. Exit status: {status}");
            }

            // Guard against a silent failure that produces a stub file: a real
            // mistralrs_quant.metallib is tens of MB; anything under 4 KB means
            // the linker ran but produced no functions.
            let metallib_size = std::fs::metadata(&metallib)
                .map(|m| m.len())
                .unwrap_or(0);
            if metallib_size < 4096 {
                panic!(
                    "Metallib sanity check failed: {path} is only {size} bytes \
                     (expected tens of MB).  The xcrun metallib step likely \
                     succeeded with an empty output.  Check that Xcode command \
                     line tools are installed and that xcrun can find the \
                     '{sdk}' SDK.\n\
                     \n\
                     Set MISTRALRS_METAL_PRECOMPILE=0 to fall back to runtime \
                     Metal compilation and skip this precompilation step.",
                    path = metallib.display(),
                    size = metallib_size,
                    sdk = platform.sdk(),
                );
            }
            println!(
                "cargo:warning=mistralrs-quant: {lib_name} linked successfully \
                 ({metallib_size} bytes)"
            );

            Ok(())
        }

        compile(Platform::MacOS)?;
        compile(Platform::Ios)?;
        compile(Platform::TvOS)?;

        Ok(())
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    Ok(())
}
