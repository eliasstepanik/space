//! The crate's own rules, checked against the manifests on disk: it depends on
//! no workspace crate, and every other crate lists it as a dev-dependency only.

use std::fs;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

#[test]
fn this_crate_depends_on_no_workspace_crate() {
    let manifest = fs::read_to_string(root().join("Cargo.toml")).expect("own Cargo.toml");
    for line in manifest.lines() {
        assert!(
            !line.starts_with("space-"),
            "space-test-support must not depend on a workspace crate: {line}"
        );
    }
}

#[test]
fn other_crates_use_it_as_a_dev_dependency_only() {
    let mut manifests: Vec<PathBuf> = fs::read_dir(root().join(".."))
        .expect("crates directory")
        .map(|e| e.expect("entry").path().join("Cargo.toml"))
        .filter(|p| p.is_file())
        .collect();
    manifests.push(root().join("../../xtask/Cargo.toml"));

    let mut checked = 0;
    for path in manifests {
        let text = fs::read_to_string(&path).expect("manifest");
        let mut section = "";
        for line in text.lines() {
            if line.starts_with('[') {
                section = line.trim();
            }
            if line.starts_with("space-test-support") {
                checked += 1;
                assert_eq!(
                    section,
                    "[dev-dependencies]",
                    "{}: `{line}` is under {section}",
                    path.display()
                );
            }
        }
    }
    assert!(checked > 0, "no manifest uses space-test-support");
}
