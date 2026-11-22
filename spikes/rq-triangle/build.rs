//! Build-time WGSL -> SPIR-V compilation via naga.
//!
//! The WGSL ray-query verdict for the pinned naga version is printed here so it
//! is captured in the build log (see README for the full story).

use std::path::Path;

/// naga does not expose its version as a constant; read it from Cargo.lock.
fn naga_version() -> String {
    std::fs::read_to_string("Cargo.lock")
        .ok()
        .and_then(|lock| {
            let idx = lock.find("name = \"naga\"")?;
            let rest = &lock[idx..];
            let v = rest.find("version = \"")? + "version = \"".len();
            let end = rest[v..].find('"')? + v;
            Some(rest[v..end].to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    let shader_path = "shaders/rq_triangle.wgsl";
    println!("cargo:rerun-if-changed={shader_path}");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:warning=naga version: {}", naga_version());

    let out_dir = std::env::var("OUT_DIR").unwrap();
    match compile(shader_path, Path::new(&out_dir).join("rq_triangle.spv")) {
        Ok(words) => println!(
            "cargo:warning=WGSL ray query: COMPILED OK ({} SPIR-V words)",
            words.len()
        ),
        Err(err) => panic!("WGSL ray query: FAILED\n{err}"),
    }
}

fn compile(shader_path: &str, out_path: std::path::PathBuf) -> Result<Vec<u32>, String> {
    let src = std::fs::read_to_string(shader_path).map_err(|e| e.to_string())?;

    let module = naga::front::wgsl::parse_str(&src)
        .map_err(|e| format!("WGSL parse error:\n{}", e.emit_to_string(&src)))?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("naga validation error: {e:?}"))?;

    // SPIR-V 1.4: RADV advertises `VK_KHR_spirv_1_4`; the ray-query extension
    // ops themselves come from SPV_KHR_ray_query, which naga emits
    // automatically when it sees ray-query builtins.
    let options = naga::back::spv::Options {
        lang_version: (1, 4),
        flags: naga::back::spv::WriterFlags::empty(),
        ..Default::default()
    };
    let pipeline_options = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };

    let words = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options))
        .map_err(|e| format!("SPIR-V write error: {e:?}"))?;

    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&out_path, &bytes).map_err(|e| e.to_string())?;
    Ok(words)
}
