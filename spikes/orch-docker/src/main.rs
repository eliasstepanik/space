//! Phase-0 spike 0.6: Docker-as-orchestrator mechanics probe.
//!
//! Validates the primitives the Phase-3 `space-orchestrator` needs:
//!
//! 1. Detect Docker reachability (`Docker::connect_with_defaults()` + `ping`).
//! 2. Create a sibling "shard" container from a tiny image (alpine pinned by
//!    digest) that sleeps and exposes a UDP port published from a configurable
//!    host-port range.
//! 3. Health-check it (container running + published host port allocated).
//! 4. List containers filtered by label `space.cluster=<id>` + `space.role=shard`.
//! 5. Remove a container by id and reap orphans (labelled containers not in the
//!    expected set).
//!
//! Every step is emitted as a JSON object on stdout (one `report` array entry
//! per step) so the run can be captured to `report-sandbox.json`.
//!
//! Standalone throwaway spike: not a member of the root `space` workspace.

use std::collections::HashMap;
use std::time::Duration;

use bollard::models::{
    ContainerCreateBody, HostConfig, PortBinding, PortMap, RestartPolicyNameEnum,
};
use bollard::query_parameters::{
    CreateContainerOptionsBuilder, InspectContainerOptions, ListContainersOptionsBuilder,
    RemoveContainerOptionsBuilder,
};
use bollard::Docker;
use futures_util::StreamExt;
use serde_json::{json, Value};

/// alpine 3.20 manifest-list digest, resolved from the Docker Hub registry API
/// on 2026-11-19 (`docker-content-digest` for `library/alpine:3.20`).
const ALPINE_BY_DIGEST: &str =
    "alpine:3.20@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc";

/// Container-internal UDP port the dummy shard "listens" on.
const SHARD_UDP_PORT: u16 = 46000;

/// Default inclusive host-port range to publish shard UDP ports from.
const DEFAULT_PORT_RANGE: (u16, u16) = (46001, 46064);

/// Label keys every shard container carries.
const LABEL_CLUSTER: &str = "space.cluster";
const LABEL_ROLE: &str = "space.role";

/// A single step in the run report.
#[derive(Debug)]
struct Step {
    name: &'static str,
    ok: bool,
    detail: Value,
    error: Option<String>,
}

impl Step {
    fn ok(name: &'static str, detail: Value) -> Self {
        Self { name, ok: true, detail, error: None }
    }
    fn fail(name: &'static str, error: impl ToString, detail: Value) -> Self {
        Self { name, ok: false, detail, error: Some(error.to_string()) }
    }
    fn to_json(&self) -> Value {
        json!({
            "step": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "error": self.error,
        })
    }
}

/// Parse a `HOST_PORT_RANGE` env var of the form `"46001-46064"`.
fn parse_port_range(raw: Option<&str>) -> Option<(u16, u16)> {
    let raw = raw?;
    let (lo, hi) = raw.split_once('-')?;
    let lo: u16 = lo.trim().parse().ok()?;
    let hi: u16 = hi.trim().parse().ok()?;
    if lo <= hi {
        Some((lo, hi))
    } else {
        None
    }
}

/// Build the label map applied to every shard container this spike creates.
fn shard_labels(cluster_id: &str) -> HashMap<String, String> {
    HashMap::from([
        (LABEL_CLUSTER.to_string(), cluster_id.to_string()),
        (LABEL_ROLE.to_string(), "shard".to_string()),
    ])
}

/// Build the `ContainerCreateBody` for a dummy shard: alpine by digest, running
/// `sleep infinity`, exposing a UDP port published from the given host range.
///
/// The published host port is requested as a fixed allocation from the range so
/// the health check can assert a specific port was allocated.
fn shard_create_body(cluster_id: &str, host_port: u16) -> ContainerCreateBody {
    let mut port_bindings = PortMap::new();
    port_bindings.insert(
        format!("{SHARD_UDP_PORT}/udp"),
        Some(vec![PortBinding {
            host_ip: Some("0.0.0.0".to_string()),
            host_port: Some(host_port.to_string()),
        }]),
    );

    let host_config = HostConfig {
        port_bindings: Some(port_bindings),
        // Never restart: the spike always cleans up explicitly.
        restart_policy: Some(bollard::models::RestartPolicy {
            name: Some(RestartPolicyNameEnum::NO),
            maximum_retry_count: None,
        }),
        ..Default::default()
    };

    ContainerCreateBody {
        image: Some(ALPINE_BY_DIGEST.to_string()),
        // `sleep infinity` keeps the container alive as a dummy shard.
        cmd: Some(vec!["sleep".to_string(), "infinity".to_string()]),
        exposed_ports: Some(vec![format!("{SHARD_UDP_PORT}/udp")]),
        labels: Some(shard_labels(cluster_id)),
        host_config: Some(host_config),
        ..Default::default()
    }
}

/// Pull an image by its full reference (name or name@digest), draining the
/// progress stream and surfacing the first error, if any.
async fn pull_image(docker: &Docker, image_ref: &str) -> Result<String, bollard::errors::Error> {
    // Per the Docker Engine API, if `fromImage` includes a digest the image is
    // pulled by digest and `tag` is ignored, so we pass the full reference.
    let options = bollard::query_parameters::CreateImageOptionsBuilder::new()
        .from_image(image_ref)
        .build();
    let mut stream = docker.create_image(Some(options), None, None);
    let mut last_status = String::new();
    while let Some(item) = stream.next().await {
        // `create_image` already maps `error_detail.message` to
        // `Error::DockerStreamError`, so any `Err` here is a pull failure.
        let info = item?;
        if let Some(status) = info.status {
            last_status = status;
        }
    }
    Ok(last_status)
}

/// Extract the allocated host port for the shard UDP port from an inspect
/// result. Returns `None` if the port is not (yet) bound.
fn allocated_host_port(inspect: &bollard::models::ContainerInspectResponse) -> Option<u16> {
    let ports = inspect.network_settings.as_ref()?.ports.as_ref()?;
    let bindings = ports.get(&format!("{SHARD_UDP_PORT}/udp"))?.as_ref()?;
    let binding = bindings.first()?;
    binding.host_port.as_ref()?.parse().ok()
}

/// Health check: container state is running AND a published host port is
/// allocated for the shard UDP port.
async fn health_check(
    docker: &Docker,
    container_id: &str,
    expected_host_port: u16,
) -> Result<Value, String> {
    let inspect = docker
        .inspect_container(container_id, None::<InspectContainerOptions>)
        .await
        .map_err(|e| format!("inspect failed: {e}"))?;

    let state = inspect.state.clone().unwrap_or_default();
    let running = state.running.unwrap_or(false);
    let status = state
        .status
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let host_port = allocated_host_port(&inspect);

    let port_ok = host_port == Some(expected_host_port);
    if running && port_ok {
        Ok(json!({
            "container_id": container_id,
            "state_status": status,
            "running": running,
            "shard_udp_port": SHARD_UDP_PORT,
            "expected_host_port": expected_host_port,
            "allocated_host_port": host_port,
        }))
    } else {
        Err(format!(
            "health check failed: running={running} status={status} \
             expected_host_port={expected_host_port} allocated_host_port={host_port:?}"
        ))
    }
}

/// List shard containers for a cluster via label filters.
async fn list_shards(
    docker: &Docker,
    cluster_id: &str,
) -> Result<Vec<bollard::models::ContainerSummary>, bollard::errors::Error> {
    let mut filters: HashMap<String, Vec<String>> = HashMap::new();
    filters.insert(
        "label".to_string(),
        vec![
            format!("{LABEL_CLUSTER}={cluster_id}"),
            format!("{LABEL_ROLE}=shard"),
        ],
    );
    let options = ListContainersOptionsBuilder::new()
        .all(true)
        .filters(&filters)
        .build();
    docker.list_containers(Some(options)).await
}

/// Remove a container by id (force + remove anonymous volumes).
async fn remove_container(docker: &Docker, container_id: &str) -> Result<(), bollard::errors::Error> {
    let options = RemoveContainerOptionsBuilder::new()
        .force(true)
        .v(true)
        .build();
    docker.remove_container(container_id, Some(options)).await
}

/// Reap orphans: labelled shard containers for `cluster_id` whose id is NOT in
/// `expected`. Returns the ids removed.
async fn reap_orphans(
    docker: &Docker,
    cluster_id: &str,
    expected: &[String],
) -> Result<Vec<String>, bollard::errors::Error> {
    let shards = list_shards(docker, cluster_id).await?;
    let mut removed = Vec::new();
    for s in shards {
        let id = match s.id {
            Some(ref id) => id.clone(),
            None => continue,
        };
        if !expected.contains(&id) {
            remove_container(docker, &id).await?;
            removed.push(id);
        }
    }
    Ok(removed)
}

#[tokio::main]
async fn main() {
    let cluster_id = std::env::var("SPACE_CLUSTER_ID")
        .unwrap_or_else(|_| "spike-0.6".to_string());
    let port_range = parse_port_range(std::env::var("HOST_PORT_RANGE").ok().as_deref())
        .unwrap_or(DEFAULT_PORT_RANGE);

    let mut steps: Vec<Step> = Vec::new();

    // ---- Step 1: detect Docker reachability -------------------------------
    let docker = match Docker::connect_with_defaults() {
        Ok(d) => d,
        Err(e) => {
            steps.push(Step::fail(
                "detect_docker",
                e,
                json!({ "connected": false }),
            ));
            emit_report(&cluster_id, port_range, &steps, false);
            return;
        }
    };

    match tokio::time::timeout(Duration::from_secs(10), docker.ping()).await {
        Ok(Ok(pong)) => {
            steps.push(Step::ok(
                "detect_docker",
                json!({
                    "connected": true,
                    "ping": pong,
                    "docker_host": std::env::var("DOCKER_HOST")
                        .unwrap_or_else(|_| "unix:///var/run/docker.sock (default)".to_string()),
                }),
            ));
        }
        Ok(Err(e)) => {
            steps.push(Step::fail("detect_docker", e, json!({ "connected": false })));
            emit_report(&cluster_id, port_range, &steps, false);
            return;
        }
        Err(_) => {
            steps.push(Step::fail(
                "detect_docker",
                "ping timed out after 10s",
                json!({ "connected": false }),
            ));
            emit_report(&cluster_id, port_range, &steps, false);
            return;
        }
    }

    // ---- Step 2: create a sibling shard container -------------------------
    let host_port = port_range.0;
    let pull_result = pull_image(&docker, ALPINE_BY_DIGEST).await;
    let pull_detail = match &pull_result {
        Ok(status) => json!({ "pulled": true, "image": ALPINE_BY_DIGEST, "last_status": status }),
        Err(e) => json!({ "pulled": false, "image": ALPINE_BY_DIGEST, "pull_error": e.to_string() }),
    };

    let shard_name = format!("space-shard-{cluster_id}");
    let created_id: Option<String> = match docker
        .create_container(
            Some(CreateContainerOptionsBuilder::default().name(&shard_name).build()),
            shard_create_body(&cluster_id, host_port),
        )
        .await
    {
        Ok(resp) => {
            let id = resp.id.clone();
            let mut detail = json!({
                "container_id": id,
                "name": shard_name,
                "image": ALPINE_BY_DIGEST,
                "published_host_port": host_port,
                "warnings": resp.warnings,
            });
            // Merge pull info into the create step detail.
            if let Some(obj) = detail.as_object_mut() {
                if let Some(p) = pull_detail.as_object() {
                    for (k, v) in p {
                        obj.insert(k.clone(), v.clone());
                    }
                }
            }
            steps.push(Step::ok("create_shard", detail));
            Some(id)
        }
        Err(e) => {
            steps.push(Step::fail("create_shard", e, pull_detail));
            None
        }
    };

    // Start it (create does not start).
    let started = match created_id.as_deref() {
        Some(id) => match docker.start_container(id, None).await {
            Ok(()) => {
                steps.push(Step::ok("start_shard", json!({ "container_id": id })));
                true
            }
            Err(e) => {
                steps.push(Step::fail("start_shard", e, json!({ "container_id": id })));
                false
            }
        },
        None => false,
    };

    // ---- Step 3: health check --------------------------------------------
    let mut healthy_id: Option<String> = None;
    if started {
        if let Some(id) = created_id.as_deref() {
            // Give the container a moment to transition to running.
            let mut last_err = String::new();
            let mut ok_detail: Option<Value> = None;
            for _ in 0..10 {
                match health_check(&docker, id, host_port).await {
                    Ok(detail) => {
                        ok_detail = Some(detail);
                        break;
                    }
                    Err(e) => {
                        last_err = e;
                        tokio::time::sleep(Duration::from_millis(500)).await;
                    }
                }
            }
            match ok_detail {
                Some(detail) => {
                    steps.push(Step::ok("health_check", detail));
                    healthy_id = Some(id.to_string());
                }
                None => {
                    steps.push(Step::fail(
                        "health_check",
                        last_err,
                        json!({ "container_id": id }),
                    ));
                }
            }
        }
    }

    // ---- Step 4: list by label -------------------------------------------
    match list_shards(&docker, &cluster_id).await {
        Ok(shards) => {
            let listed: Vec<Value> = shards
                .iter()
                .map(|s| {
                    json!({
                        "id": s.id,
                        "names": s.names,
                        "state": s.state.as_ref().map(|x| x.to_string()),
                        "labels": s.labels,
                    })
                })
                .collect();
            steps.push(Step::ok(
                "list_shards",
                json!({
                    "cluster_id": cluster_id,
                    "count": listed.len(),
                    "filters": {
                        "label": [
                            format!("{LABEL_CLUSTER}={cluster_id}"),
                            format!("{LABEL_ROLE}=shard"),
                        ],
                    },
                    "containers": listed,
                }),
            ));
        }
        Err(e) => {
            steps.push(Step::fail("list_shards", e, json!({ "cluster_id": cluster_id })));
        }
    }

    // ---- Step 5: remove by id + reap orphans -----------------------------
    // Remove the shard we created (by id), then reap any remaining orphans
    // (labelled shards not in the expected set — here the expected set is empty
    // after removal, so reaping should clean anything left over).
    let mut removed_ids: Vec<String> = Vec::new();
    let mut remove_ok = true;
    if let Some(id) = healthy_id.as_deref().or(created_id.as_deref()) {
        match remove_container(&docker, id).await {
            Ok(()) => {
                removed_ids.push(id.to_string());
            }
            Err(e) => {
                remove_ok = false;
                steps.push(Step::fail("remove_shard", e, json!({ "container_id": id })));
            }
        }
    }
    if remove_ok {
        steps.push(Step::ok(
            "remove_shard",
            json!({ "removed_ids": removed_ids, "force": true, "volumes": true }),
        ));
    }

    // Reap orphans: expected set is empty (we removed ours), so any labelled
    // shard left for this cluster is an orphan.
    match reap_orphans(&docker, &cluster_id, &[]).await {
        Ok(orphan_ids) => {
            steps.push(Step::ok(
                "reap_orphans",
                json!({
                    "cluster_id": cluster_id,
                    "expected_ids": [],
                    "reaped_ids": orphan_ids,
                }),
            ));
        }
        Err(e) => {
            steps.push(Step::fail(
                "reap_orphans",
                e,
                json!({ "cluster_id": cluster_id }),
            ));
        }
    }

    let all_ok = steps.iter().all(|s| s.ok);
    emit_report(&cluster_id, port_range, &steps, all_ok);
}

/// Print the final JSON report to stdout.
fn emit_report(cluster_id: &str, port_range: (u16, u16), steps: &[Step], all_ok: bool) {
    let report = json!({
        "spike": "0.6",
        "cluster_id": cluster_id,
        "image": ALPINE_BY_DIGEST,
        "shard_udp_port": SHARD_UDP_PORT,
        "host_port_range": [port_range.0, port_range.1],
        "bollard_version": "0.21.1",
        "bollard_stubs": "1.53.1-rc.29.3.1",
        "all_steps_ok": all_ok,
        "steps": steps.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard::models::{ContainerInspectResponse, NetworkSettings};

    // ---- parse_port_range -------------------------------------------------
    #[test]
    fn parse_port_range_valid() {
        assert_eq!(parse_port_range(Some("46001-46064")), Some((46001, 46064)));
    }

    #[test]
    fn parse_port_range_trims_whitespace() {
        assert_eq!(parse_port_range(Some(" 46001 - 46064 ")), Some((46001, 46064)));
    }

    #[test]
    fn parse_port_range_single_port() {
        assert_eq!(parse_port_range(Some("46001-46001")), Some((46001, 46001)));
    }

    #[test]
    fn parse_port_range_reversed_is_none() {
        assert_eq!(parse_port_range(Some("46064-46001")), None);
    }

    #[test]
    fn parse_port_range_garbage_is_none() {
        assert_eq!(parse_port_range(None), None);
        assert_eq!(parse_port_range(Some("")), None);
        assert_eq!(parse_port_range(Some("nope")), None);
        assert_eq!(parse_port_range(Some("abc-def")), None);
        assert_eq!(parse_port_range(Some("70000-80000")), None); // > u16
    }

    // ---- shard_labels -----------------------------------------------------
    #[test]
    fn shard_labels_has_cluster_and_role() {
        let labels = shard_labels("cluster-xyz");
        assert_eq!(labels.get(LABEL_CLUSTER).map(String::as_str), Some("cluster-xyz"));
        assert_eq!(labels.get(LABEL_ROLE).map(String::as_str), Some("shard"));
        assert_eq!(labels.len(), 2);
    }

    // ---- shard_create_body ------------------------------------------------
    #[test]
    fn shard_create_body_pins_image_and_cmd() {
        let body = shard_create_body("c1", 46001);
        assert_eq!(body.image.as_deref(), Some(ALPINE_BY_DIGEST));
        assert!(ALPINE_BY_DIGEST.contains("@sha256:"));
        assert_eq!(body.cmd, Some(vec!["sleep".to_string(), "infinity".to_string()]));
    }

    #[test]
    fn shard_create_body_exposes_udp_and_binds_host_port() {
        let body = shard_create_body("c1", 46010);
        assert_eq!(body.exposed_ports, Some(vec![format!("{SHARD_UDP_PORT}/udp")]));

        let hc = body.host_config.expect("host_config present");
        let port_bindings = hc.port_bindings.expect("port_bindings present");
        let bindings = port_bindings
            .get(&format!("{SHARD_UDP_PORT}/udp"))
            .expect("udp key present")
            .as_ref()
            .expect("binding list present");
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].host_port.as_deref(), Some("46010"));
        assert_eq!(bindings[0].host_ip.as_deref(), Some("0.0.0.0"));
    }

    #[test]
    fn shard_create_body_sets_labels_and_no_restart() {
        let body = shard_create_body("c1", 46001);
        let labels = body.labels.expect("labels present");
        assert_eq!(labels.get(LABEL_CLUSTER).map(String::as_str), Some("c1"));
        assert_eq!(labels.get(LABEL_ROLE).map(String::as_str), Some("shard"));

        let hc = body.host_config.expect("host_config present");
        let rp = hc.restart_policy.expect("restart_policy present");
        assert_eq!(rp.name, Some(RestartPolicyNameEnum::NO));
    }

    // ---- allocated_host_port ----------------------------------------------
    fn inspect_with_port(port: Option<&str>) -> ContainerInspectResponse {
        let mut port_map = PortMap::new();
        let bindings = port.map(|p| {
            vec![PortBinding { host_ip: Some("0.0.0.0".into()), host_port: Some(p.into()) }]
        });
        port_map.insert(format!("{SHARD_UDP_PORT}/udp"), bindings);
        ContainerInspectResponse {
            network_settings: Some(NetworkSettings {
                ports: Some(port_map),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn allocated_host_port_reads_bound_port() {
        assert_eq!(allocated_host_port(&inspect_with_port(Some("46010"))), Some(46010));
    }

    #[test]
    fn allocated_host_port_none_when_unbound() {
        assert_eq!(allocated_host_port(&inspect_with_port(None)), None);
    }

    #[test]
    fn allocated_host_port_none_when_no_network_settings() {
        assert_eq!(allocated_host_port(&ContainerInspectResponse::default()), None);
    }

    // ---- Step JSON shape --------------------------------------------------
    #[test]
    fn step_ok_and_fail_json_shape() {
        let ok = Step::ok("s", json!({ "k": 1 }));
        let ok_json = ok.to_json();
        assert_eq!(ok_json["step"], "s");
        assert_eq!(ok_json["ok"], true);
        assert_eq!(ok_json["error"], Value::Null);

        let bad = Step::fail("s", "boom", json!({}));
        let bad_json = bad.to_json();
        assert_eq!(bad_json["ok"], false);
        assert_eq!(bad_json["error"], "boom");
    }
}
