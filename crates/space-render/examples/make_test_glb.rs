//! Write a built-in sample model as a `.glb` file, so the GLB path can be
//! tried without any external asset: the ship (the default) or, with
//! `--character`, the placeholder player character. Both carry their
//! procedural textures as embedded PNGs.
//!
//! ```sh
//! cargo run -p space-render --example make_test_glb -- ship.glb
//! cargo run -p space-render --example make_test_glb -- --character character.glb
//! cargo run -p space-client -- --window --materials --glb ship.glb
//! ```

use std::path::PathBuf;

use space_render::{sample_character, sample_ship, MeshData};

fn main() {
    let mut character = false;
    let mut path: Option<PathBuf> = None;
    for arg in std::env::args_os().skip(1) {
        if arg == "--character" {
            character = true;
        } else if arg == "--ship" {
            character = false;
        } else if arg.to_string_lossy().starts_with("--") || path.is_some() {
            eprintln!("usage: make_test_glb [--ship | --character] [FILE.glb]");
            std::process::exit(2);
        } else {
            path = Some(PathBuf::from(arg));
        }
    }
    let (default_name, mesh) = if character {
        ("character.glb", sample_character())
    } else {
        ("ship.glb", sample_ship())
    };
    let path = path.unwrap_or_else(|| PathBuf::from(default_name));
    let bytes = mesh.to_glb();
    if let Err(e) = std::fs::write(&path, &bytes) {
        eprintln!("make_test_glb: {}: {e}", path.display());
        std::process::exit(1);
    }
    // Read it back through the loader: the file on disk is known good.
    match MeshData::load(&path) {
        Ok(loaded) => eprintln!(
            "make_test_glb: wrote {} ({} bytes, {} triangles, {} material parts, {} textures)",
            path.display(),
            bytes.len(),
            loaded.triangle_count(),
            loaded.parts.len(),
            loaded.textures.len()
        ),
        Err(e) => {
            eprintln!("make_test_glb: the written file does not load: {e}");
            std::process::exit(1);
        }
    }
}
