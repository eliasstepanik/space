//! The orchestrator's startup wiring, shared by the binary and by tests.
//!
//! [`boot`] opens the persistent id counter, pre-allocates the root owner's id,
//! builds the epoch-1 map and the [`ControlPlane`], and marks that root owner as
//! *unclaimed*: the orchestrator reserves the id (so a restart resumes above it,
//! H9) but launches no shard for it, so the first live shard that asks for the
//! root adopts it ([`ControlPlane::handle_shard_hello`]).
//!
//! It settles the **cluster secret** too: a control port that is not on
//! loopback does not start without one a shard could present
//! ([`OrchestratorConfig::check_secret`]), and when the operator configured
//! none, one is generated for the shards the orchestrator starts itself (it is
//! handed to them by the launcher only when a hello needs it, and never
//! printed).
//!
//! It also opens the scene file next to the counter ([`crate::scene`]): the
//! scene an operator chose before a restart is restored, otherwise the
//! configured initial scene applies. Unlike the counter, a scene file that
//! cannot be used never fails the boot — it is reported and the initial scene
//! is used.

use space_protocol::shard::{ShardMap, ShardRegion};

use crate::config::{OrchestratorConfig, Secret};
use crate::control::ControlPlane;
use crate::epoch::EpochPublisher;
use crate::ids::PersistentShardIdAllocator;
use crate::scene::{scene_path, SceneStore};

/// Build the control plane the binary runs.
///
/// The initial map holds one region at the configured root, owned by the first
/// allocated id and carrying the control port as a placeholder endpoint (the
/// adopting shard replaces it with its own address on its hello).
///
/// # Errors
///
/// The control port is not on loopback and no shard could present a cluster
/// secret ([`OrchestratorConfig::check_secret`]); `min_shards` is more than
/// the cluster can hold ([`OrchestratorConfig::check_shard_counts`]). The id
/// counter file is corrupt, out of range or unreadable, the id space is
/// exhausted, or the counter cannot be **written** (a `--state` path in a
/// directory that does not exist, a read-only working directory): that is an
/// error naming the path, not a panic.
pub fn boot(config: OrchestratorConfig, peer_token: String) -> Result<ControlPlane, String> {
    config.check_secret()?;
    config.check_shard_counts()?;
    let mut ids = PersistentShardIdAllocator::open(&config.state_path, config.first_shard_id)
        .map_err(|e| e.to_string())?;
    let root_owner = ids
        .allocate()
        .map_err(|e| format!("{}: {e}", config.state_path.display()))?;
    let map = ShardMap::new(
        1,
        vec![ShardRegion::new(
            config.root_node(),
            root_owner,
            config.bind,
        )],
    );
    let scene = SceneStore::open(&scene_path(&config.state_path), config.initial_scene);
    if let Some(warning) = scene.warning() {
        // Not an error: see the module docs. The orchestrator also puts it in
        // its event log, so it shows on the operator's page.
        eprintln!("space-orchestrator: {warning}");
    }
    if let Some(note) = scene.note() {
        // A retired scene replaced by the default: expected, not a problem.
        println!("space-orchestrator: {note}");
    }
    let mut control = ControlPlane::new(config, ids, EpochPublisher::new(map), peer_token);
    control.scene = scene;
    control.set_unclaimed_root_owner(root_owner);
    // No secret configured: make one for the shards this orchestrator starts
    // itself (a configured one is kept). From the same source as the web UI's
    // form token; whether a hello *needs* it is the config's rule.
    control.set_generated_secret(Secret::new(crate::webui::auth::random_token()));
    Ok(control)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        space_test_support::temp_path("space-orch-boot-unit", tag)
    }

    /// Mutation this catches: the `expect` that used to sit on the counter
    /// write — `space-orchestrator --state <dir that does not exist>/x.id`
    /// panicked at startup instead of printing why it could not start.
    #[test]
    fn an_unwritable_state_path_is_a_boot_error_naming_the_path() {
        let dir = temp_path("missing-dir");
        let _ = std::fs::remove_dir_all(&dir);
        let state_path = dir.join("orchestrator.id");
        let config = OrchestratorConfig {
            state_path: state_path.clone(),
            ..OrchestratorConfig::default()
        };
        let err = boot(config, String::new()).expect_err("the counter cannot be written");
        assert!(
            err.contains(&state_path.display().to_string()),
            "the error names the state path: {err}"
        );
        assert!(err.contains("could not be persisted"), "{err}");
    }

    /// The cluster secret after a boot: the configured one, else a generated
    /// one — and a control port that is not on loopback does not boot when no
    /// shard could present one.
    ///
    /// Mutation this catches: booting an open control port with no secret at
    /// all, generating over a configured secret, or generating the same one
    /// twice.
    #[test]
    fn boot_settles_the_cluster_secret() {
        let fresh = |tag: &str| {
            let state_path = temp_path(tag).with_extension("id");
            let _ = std::fs::remove_file(&state_path);
            OrchestratorConfig {
                state_path,
                ..OrchestratorConfig::default()
            }
        };
        // Loopback, nothing configured: one is generated for launched shards,
        // and a hello needs none.
        let a = boot(fresh("secret-a"), String::new()).expect("boot");
        let b = boot(fresh("secret-b"), String::new()).expect("boot");
        assert!(!a.secret_required() && !b.secret_required());
        let generated = a.cluster_secret().expose().to_owned();
        assert!(generated.len() >= OrchestratorConfig::MIN_SECRET_LEN);
        assert_ne!(
            generated,
            b.cluster_secret().expose(),
            "one per orchestrator"
        );

        // A configured secret is the secret, and is required.
        let configured = OrchestratorConfig {
            cluster_secret: Some(Secret::new("configured-cluster-secret")),
            ..fresh("secret-c")
        };
        let c = boot(configured, String::new()).expect("boot");
        assert!(c.secret_required());
        assert_eq!(c.cluster_secret().expose(), "configured-cluster-secret");

        // A reachable control port with no secret anybody could present.
        let open = OrchestratorConfig {
            bind: "0.0.0.0:7900".parse().expect("an address"),
            ..fresh("secret-d")
        };
        let state = open.state_path.clone();
        let err = boot(open.clone(), String::new()).expect_err("must not start");
        assert!(err.contains("without a cluster secret"), "{err}");
        assert!(!state.exists(), "refused before anything was written");
        // ...but fine when it starts the shards itself: they get the generated one.
        let managed = boot(
            OrchestratorConfig {
                min_shards: 4,
                ..open
            },
            String::new(),
        )
        .expect("boot");
        assert!(managed.secret_required());
        assert!(!managed.cluster_secret().is_empty());
        for control in [a, b, c, managed] {
            std::fs::remove_file(&control.config.state_path).ok();
        }
    }

    #[test]
    fn boot_reserves_the_root_owner_and_persists_the_counter() {
        let state_path = temp_path("ok").with_extension("id");
        let _ = std::fs::remove_file(&state_path);
        let config = OrchestratorConfig {
            state_path: state_path.clone(),
            ..OrchestratorConfig::default()
        };
        let control = boot(config.clone(), String::new()).expect("boot");
        let owner = control.unclaimed_root_owner().expect("a root owner");
        assert_eq!(u64::from(owner.0), config.first_shard_id);
        assert_eq!(control.map().regions[0].shard, owner);
        // A second boot over the same file resumes strictly above it (H9).
        drop(control);
        let again = boot(config, String::new()).expect("second boot");
        assert!(again.unclaimed_root_owner().expect("a root owner") > owner);
        std::fs::remove_file(&state_path).ok();
    }

    /// The scene is opened next to the counter: the configured initial scene
    /// on a fresh state directory, the saved one after a restart — and a scene
    /// file that makes no sense is a warning, never a refusal to boot.
    ///
    /// Mutation this catches: `?` on the scene file like on the counter (one
    /// stray byte would keep the whole control plane down), or booting with the
    /// in-memory store (a restart forgets the operator's choice).
    #[test]
    fn boot_restores_a_saved_scene_and_survives_a_corrupt_scene_file() {
        use space_protocol::{Scene, SceneState};
        let state_path = temp_path("scene").with_extension("id");
        let scene_file = scene_path(&state_path);
        let _ = std::fs::remove_file(&state_path);
        let _ = std::fs::remove_file(&scene_file);
        let config = OrchestratorConfig {
            state_path: state_path.clone(),
            initial_scene: Scene::Materials,
            ..OrchestratorConfig::default()
        };

        let mut control = boot(config.clone(), String::new()).expect("boot");
        assert_eq!(
            control.scene.current(),
            SceneState::new(Scene::Materials.id(), 1),
            "the initial scene, since nothing was saved"
        );
        assert_eq!(control.scene.path(), Some(scene_file.as_path()));
        control.scene.set(Scene::Galaxy).expect("saved");
        drop(control);

        // Restarted with the same flags: the saved scene wins over `--scene`.
        let control = boot(config.clone(), String::new()).expect("second boot");
        assert_eq!(
            control.scene.current(),
            SceneState::new(Scene::Galaxy.id(), 2)
        );
        assert!(control.scene.restored());
        drop(control);

        // The file is damaged: the orchestrator still starts.
        std::fs::write(&scene_file, b"\xff\xfe not a scene").expect("damage it");
        let control = boot(config, String::new()).expect("a bad scene file never stops a boot");
        assert_eq!(
            control.scene.current(),
            SceneState::new(Scene::Materials.id(), 1)
        );
        assert!(control.scene.warning().is_some());
        std::fs::remove_file(&state_path).ok();
        std::fs::remove_file(&scene_file).ok();
    }
}
