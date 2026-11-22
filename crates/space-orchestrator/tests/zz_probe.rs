use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use space_orchestrator::boot::boot;
use space_orchestrator::config::OrchestratorConfig;
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::orchestrator::Orchestrator;
use space_protocol::ShardId;
use space_server::orchestrated::{start_orchestrated, OrchestratedOptions};

fn l0() -> SocketAddr { SocketAddr::from(([127,0,0,1],0)) }

#[tokio::test]
async fn probe_two_shards_and_rehello() {
    let sp = std::env::temp_dir().join(format!("zzprobe-{}.id", std::process::id()));
    let _ = std::fs::remove_file(&sp);
    let config = OrchestratorConfig { bind: l0(), state_path: sp.clone(), ..OrchestratorConfig::default() };
    let cluster = config.cluster.clone();
    let control = boot(config, "secret".into()).unwrap();
    let mut o = Orchestrator::new(control, Arc::new(LocalLauncher::new()));
    let server = o.bind().unwrap();
    let ca = server.local_addr().unwrap();
    tokio::spawn(async move { o.run_with(server).await });
    let opts = || OrchestratedOptions { shard: ShardId::NONE, bind: l0(), advertise: None, claim_region: None, cluster: cluster.clone() };
    let a = start_orchestrated(opts(), ca).await;
    println!("A: {:?}", a.as_ref().map(|s| s.shard).map_err(|e| e.clone()));
    let t = std::time::Instant::now();
    let b = tokio::time::timeout(Duration::from_secs(60), start_orchestrated(opts(), ca)).await;
    println!("B (distinct endpoint, no region): {:?} after {:?}", b.as_ref().map(|r| r.as_ref().map(|s| s.shard).map_err(|e| e.clone())), t.elapsed());
    // re-hello same node, same NONE (restart of shard A)
    a.unwrap().abort();
    let t = std::time::Instant::now();
    let a2 = tokio::time::timeout(Duration::from_secs(60), start_orchestrated(opts(), ca)).await;
    println!("A restart: {:?} after {:?}", a2.as_ref().map(|r| r.as_ref().map(|s| s.shard).map_err(|e| e.clone())), t.elapsed());
    let _ = std::fs::remove_file(&sp);
}
