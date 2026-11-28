//! Build-time WGSL -> SPIR-V compilation via naga (Phase 4 requirement, P2).
//!
//! `shaders/common.wgsl` is prepended to every shader (shared `Instance`
//! struct + hash helpers), and each entry point is compiled to its own SPIR-V
//! module in `OUT_DIR`. A shader file can be compiled in several *variants*:
//! lines between `//#if NAME` and `//#endif` (see [`preprocess`]) are kept
//! only for the variants that define `NAME`, which is how the forward pass
//! gets its ray-traced-shadow and mesh versions without the plain version
//! containing a line of either (`common.wgsl` is filtered the same way). naga >= 30 is required for the ray-query WGSL
//! (`enable wgpu_ray_query;`) — see spikes/rq-triangle/README.md; the version
//! is asserted here (see [`assert_naga_at_least_30`]) so the guard fires both
//! locally and in CI, not only in a log line.
//!
//! After parsing each WGSL module this build script also **derives the memory
//! layout of every bound uniform/storage block from naga itself** and writes it
//! to `$OUT_DIR/wgsl_layout.rs` (see [`dump_layout`]). `src/params.rs` includes
//! that file in its test module and asserts the Rust `offset_of!`/`size_of!`
//! values against the *generated* constants, so WGSL-side layout drift (e.g. a
//! scalar pad reverted to `vec3`) fails a test instead of silently moving a
//! member.

use std::path::Path;

const COMPUTE: naga::ShaderStage = naga::ShaderStage::Compute;
const VERTEX: naga::ShaderStage = naga::ShaderStage::Vertex;
const FRAGMENT: naga::ShaderStage = naga::ShaderStage::Fragment;

/// `(shader file, output name, shader stage, entry point, defines)`.
#[rustfmt::skip]
const SHADERS: &[(&str, &str, naga::ShaderStage, &str, &[&str])] = &[
    ("rq_triangle.wgsl", "rq_triangle", COMPUTE, "main", &[]),
    ("gi_probe.wgsl", "gi_probe", COMPUTE, "main", &["MESH", "ENV"]),
    ("tonemap.wgsl", "tonemap", COMPUTE, "main", &[]),
    ("forward.wgsl", "forward_vs", VERTEX, "vs_main", &["ENV"]),
    ("forward.wgsl", "forward_fs", FRAGMENT, "fs_main", &["ENV"]),
    ("forward.wgsl", "forward_fs_shadow", FRAGMENT, "fs_main", &["ENV", "SHADOWS"]),
    ("forward.wgsl", "forward_glass_vs", VERTEX, "vs_glass", &["ENV"]),
    ("forward.wgsl", "forward_glass_fs", FRAGMENT, "fs_glass", &["ENV"]),
    ("forward.wgsl", "forward_glass_fs_shadow", FRAGMENT, "fs_glass", &["ENV", "SHADOWS"]),
    ("forward.wgsl", "forward_mesh_vs", VERTEX, "vs_mesh", &["ENV", "MESH"]),
    ("forward.wgsl", "forward_mesh_fs", FRAGMENT, "fs_mesh", &["ENV", "MESH"]),
    ("forward.wgsl", "forward_mesh_fs_shadow", FRAGMENT, "fs_mesh", &["ENV", "MESH", "SHADOWS"]),
    ("forward.wgsl", "forward_sky_vs", VERTEX, "vs_sky", &["ENV", "SKY"]),
    ("forward.wgsl", "forward_sky_fs", FRAGMENT, "fs_sky", &["ENV", "SKY"]),
    ("post.wgsl", "post", COMPUTE, "main", &["ENV"]),
    // The same with ray queries: glass refracted through its real shape.
    ("post.wgsl", "post_rq", COMPUTE, "main", &["ENV", "MESH", "RQ"]),
    ("overlay.wgsl", "overlay_vs", VERTEX, "vs_overlay", &[]),
    ("overlay.wgsl", "overlay_fs", FRAGMENT, "fs_overlay", &[]),
];

/// `(shader output name, WGSL struct name, Rust mirror name)`.
///
/// The layout of each listed struct is dumped from naga into
/// `$OUT_DIR/wgsl_layout.rs` as `<RUST>_SIZE` / `<RUST>_<MEMBER>` constants,
/// which `src/params.rs`' tests assert the Rust layout against.
const LAYOUT_MAP: &[(&str, &str, &str)] = &[
    ("rq_triangle", "Params", "RqParams"),
    ("gi_probe", "GiParams", "GiParams"),
    ("tonemap", "ToneParams", "ToneParams"),
    ("post", "PostParams", "PostParams"),
    ("forward_fs", "Params", "ForwardParams"),
    ("forward_fs", "Camera", "Camera"),
    ("forward_fs", "Light", "Light"),
    ("forward_fs", "Instance", "GpuInstance"),
    ("forward_fs", "Environment", "EnvironmentGpu"),
    ("forward_mesh_vs", "MeshInstance", "MeshInstanceGpu"),
];

/// The `naga` version this build actually resolved, read from `Cargo.lock`.
fn naga_version() -> String {
    std::fs::read_to_string("../../Cargo.lock")
        .or_else(|_| std::fs::read_to_string("Cargo.lock"))
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

/// Reject naga < 30 (spikes/rq-triangle/README.md: naga 24 cannot even parse
/// `enable wgpu_ray_query;`, and its ray-query SPIR-V was observed to drop
/// hits). Panicking here enforces the constraint locally *and* in CI, because
/// `build.rs` runs on every build of `space-render`.
fn assert_naga_at_least_30(version: &str) {
    let major = version
        .split('.')
        .next()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or_else(|| {
            panic!(
                "cannot determine the naga version (read {version:?} from Cargo.lock); \
                 space-render requires naga >= 30 — see spikes/rq-triangle/README.md"
            )
        });
    assert!(
        major >= 30,
        "space-render requires naga >= 30 (found {version}); naga 24 cannot compile the \
         ray-query WGSL — see spikes/rq-triangle/README.md"
    );
}

/// Round `n` up to a multiple of `a`.
const fn round_up(n: u32, a: u32) -> u32 {
    n.div_ceil(a) * a
}

/// Emit the layout of every struct in [`LAYOUT_MAP`] that `module` defines, as
/// Rust `usize` constants appended to `out`.
///
/// Offsets come from naga's WGSL front end (`StructMember::offset`, which is
/// `round_up(previous_end, member_alignment)` per WGSL §Memory Layout) and the
/// block size from the struct's `span`. A struct bound directly as a `uniform`
/// block additionally has its size rounded up to a multiple of 16 — that is the
/// size the uniform buffer must be bound with.
fn dump_layout(module: &naga::Module, out_name: &str, out: &mut String) -> Result<(), String> {
    // Which struct handles are bound directly as a uniform block.
    let uniform_handles: Vec<naga::Handle<naga::Type>> = module
        .global_variables
        .iter()
        .filter(|(_, v)| v.space == naga::AddressSpace::Uniform)
        .map(|(_, v)| v.ty)
        .collect();

    for (wgsl_struct, rust_struct) in LAYOUT_MAP
        .iter()
        .filter(|(shader, _, _)| *shader == out_name)
        .map(|(_, wgsl, rust)| (*wgsl, *rust))
    {
        let handle = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some(wgsl_struct))
            .map(|(h, _)| h)
            .ok_or_else(|| format!("{out_name}: no struct named {wgsl_struct} in scope"))?;
        let uniform = uniform_handles.contains(&handle);
        dump_struct(module, handle, rust_struct, uniform, out)?;
    }
    Ok(())
}

/// Emit `<RUST>_SIZE` plus one `<RUST>_<MEMBER>` constant per member.
fn dump_struct(
    module: &naga::Module,
    handle: naga::Handle<naga::Type>,
    rust_struct: &str,
    uniform: bool,
    out: &mut String,
) -> Result<(), String> {
    let ty = &module.types[handle];
    let naga::TypeInner::Struct { ref members, span } = ty.inner else {
        return Err(format!(
            "{}: not a struct",
            ty.name.clone().unwrap_or_default()
        ));
    };
    let size = if uniform { round_up(span, 16) } else { span };
    let prefix = rust_struct.to_uppercase();
    out.push_str(&format!(
        "/// `{}` in this module's WGSL ({size} bytes).\n",
        ty.name.clone().unwrap_or_default()
    ));
    out.push_str(&format!(
        "#[allow(dead_code)]\npub const {prefix}_SIZE: usize = {size};\n"
    ));
    for member in members.iter() {
        let name = member
            .name
            .clone()
            .unwrap_or_else(|| panic!("{rust_struct}: anonymous member"))
            .to_uppercase();
        out.push_str(&format!(
            "#[allow(dead_code)]\npub const {prefix}_{name}: usize = {};",
            member.offset
        ));
        out.push('\n');
    }
    Ok(())
}

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=build.rs");
    // The lock file `naga_version` reads: the workspace's, two directories up.
    // It must be a file that exists: cargo treats a watched path that is
    // missing as changed, and a bare `Cargo.lock` (this crate's directory has
    // none) made every build recompile this crate and all that depend on it.
    let lock = ["../../Cargo.lock", "Cargo.lock"]
        .into_iter()
        .find(|path| Path::new(path).is_file());
    if let Some(lock) = lock {
        println!("cargo:rerun-if-changed={lock}");
    }
    let version = naga_version();
    assert_naga_at_least_30(&version);
    println!("cargo:warning=naga version: {version}");

    let common = std::fs::read_to_string("shaders/common.wgsl").expect("read common.wgsl");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let mut layout = String::from(
        "// @generated by build.rs from the WGSL in `shaders/` (naga layout dump).\n\n",
    );
    for (file, out_name, stage, entry, defines) in SHADERS {
        println!("cargo:rerun-if-changed=shaders/{file}");
        let src = std::fs::read_to_string(Path::new("shaders").join(file))
            .unwrap_or_else(|e| panic!("read {file}: {e}"));
        let src = preprocess(&src, defines).unwrap_or_else(|e| panic!("{file}: {e}"));
        let shared = preprocess(&common, defines).unwrap_or_else(|e| panic!("common.wgsl: {e}"));
        // `enable` directives must stay first, so insert the shared code after
        // any leading enable lines rather than blindly prepending.
        let full = splice_common(&shared, &src);
        std::fs::write(Path::new(&out_dir).join(format!("{out_name}.wgsl")), &full).ok();
        match compile_and_dump(&full, &out_dir, out_name, *stage, entry, &mut layout) {
            Ok(words) => println!("cargo:warning={out_name}: {words} SPIR-V words"),
            Err(err) => panic!("{out_name}: {err}"),
        }
    }
    std::fs::write(Path::new(&out_dir).join("wgsl_layout.rs"), layout)
        .expect("write wgsl_layout.rs");
}

/// Resolve the conditional blocks of a shader for one variant: a line that is
/// exactly `//#if NAME` starts a block kept only when `defines` contains
/// `NAME`, `//#else` switches to the opposite, `//#endif` ends it. Blocks do
/// not nest. The directive lines themselves are dropped.
fn preprocess(src: &str, defines: &[&str]) -> Result<String, String> {
    let mut out = String::with_capacity(src.len());
    // `Some(keep)` inside a block.
    let mut block: Option<bool> = None;
    for (number, line) in src.lines().enumerate() {
        let directive = line.trim();
        let at = |what: &str| format!("line {}: {what}", number + 1);
        if let Some(name) = directive.strip_prefix("//#if ") {
            if block.is_some() {
                return Err(at("nested //#if"));
            }
            block = Some(defines.contains(&name.trim()));
        } else if directive == "//#else" {
            block = Some(!block.ok_or_else(|| at("//#else without //#if"))?);
        } else if directive == "//#endif" {
            block.take().ok_or_else(|| at("//#endif without //#if"))?;
        } else if directive.starts_with("//#") {
            return Err(at("unknown directive"));
        } else if block != Some(false) {
            out.push_str(line);
            out.push('\n');
        }
    }
    match block {
        None => Ok(out),
        Some(_) => Err("unterminated //#if".to_string()),
    }
}

/// Insert `common` after the leading comment/`enable` lines of `src`.
fn splice_common(common: &str, src: &str) -> String {
    let mut lines: Vec<&str> = src.lines().collect();
    let mut insert_at = 0;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with("enable ") {
            insert_at = i + 1;
        } else {
            let _ = i;
            break;
        }
    }
    let mut out = String::new();
    for line in lines.drain(..insert_at) {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(common);
    out.push('\n');
    for line in lines {
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn compile_and_dump(
    src: &str,
    out_dir: &str,
    out_name: &str,
    stage: naga::ShaderStage,
    entry: &str,
    layout: &mut String,
) -> Result<usize, String> {
    let module = naga::front::wgsl::parse_str(src)
        .map_err(|e| format!("WGSL parse error:\n{}", e.emit_to_string(src)))?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("naga validation error: {e:?}"))?;

    dump_layout(&module, out_name, layout)?;

    // SPIR-V 1.4: RADV advertises VK_KHR_spirv_1_4; lavapipe supports it too.
    let options = naga::back::spv::Options {
        lang_version: (1, 4),
        flags: naga::back::spv::WriterFlags::empty(),
        ..Default::default()
    };
    let pipeline_options = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry.to_string(),
    };

    let words = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options))
        .map_err(|e| format!("SPIR-V write error: {e:?}"))?;

    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(Path::new(out_dir).join(format!("{out_name}.spv")), &bytes)
        .map_err(|e| format!("write {out_name}.spv: {e}"))?;
    Ok(words.len())
}
