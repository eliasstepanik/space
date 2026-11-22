//! Write the built-in sample model as a `.glb` file, so the GLB path can be
//! tried without any external asset:
//!
//! ```sh
//! cargo run -p space-render --example make_test_glb -- ship.glb
//! cargo run -p space-client -- --window --materials --glb ship.glb
//! ```

use std::path::PathBuf;

use space_render::{sample_ship, MeshData};

fn main() {
    let path = std::env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from("ship.glb"), PathBuf::from);
    let ship = sample_ship();
    let bytes = ship.to_glb();
    if let Err(e) = std::fs::write(&path, &bytes) {
        eprintln!("make_test_glb: {}: {e}", path.display());
        std::process::exit(1);
    }
    // Read it back through the loader: the file on disk is known good.
    match MeshData::load(&path) {
        Ok(loaded) => eprintln!(
            "make_test_glb: wrote {} ({} bytes, {} triangles, {} material parts)",
            path.display(),
            bytes.len(),
            loaded.triangle_count(),
            loaded.parts.len()
        ),
        Err(e) => {
            eprintln!("make_test_glb: the written file does not load: {e}");
            std::process::exit(1);
        }
    }
}
