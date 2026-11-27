//! Shared by this crate's integration tests (`mod common;`). Not a test target of its own.
//! Helpers that need this crate's types live here, not in `space-test-support` (which depends
//! on no workspace crate).
#![allow(dead_code)] // each test binary uses a different part

use std::net::SocketAddr;
use std::sync::Mutex;

use space_orchestrator::launcher::{LaunchError, LaunchId, Launcher, Poll, ShardHandle, SpawnSpec};

/// A launcher that records the orchestrator's launch decisions and never starts a process.
#[derive(Debug)]
pub struct RecordingLauncher {
    spawned: Mutex<Vec<SpawnSpec>>,
    killed: Mutex<Vec<LaunchId>>,
    process: &'static str,
    endpoint: fn(&SpawnSpec) -> SocketAddr,
    dead_once_killed: bool,
}

impl RecordingLauncher {
    pub fn new(
        process: &'static str,
        endpoint: fn(&SpawnSpec) -> SocketAddr,
        dead_once_killed: bool,
    ) -> Self {
        Self {
            spawned: Mutex::new(Vec::new()),
            killed: Mutex::new(Vec::new()),
            process,
            endpoint,
            dead_once_killed,
        }
    }

    pub fn spawned(&self) -> Vec<SpawnSpec> {
        self.spawned.lock().unwrap().clone()
    }

    pub fn killed(&self) -> Vec<LaunchId> {
        self.killed.lock().unwrap().clone()
    }
}

impl Launcher for RecordingLauncher {
    async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
        self.spawned.lock().unwrap().push(spec.clone());
        Ok(ShardHandle {
            launch: spec.launch,
            shard: None,
            endpoint: (self.endpoint)(&spec),
            process: self.process.into(),
        })
    }

    async fn kill(&self, launch: LaunchId) -> Result<(), LaunchError> {
        self.killed.lock().unwrap().push(launch);
        Ok(())
    }

    async fn poll(&self, launch: LaunchId) -> Result<Poll, LaunchError> {
        Ok(
            if self.dead_once_killed && self.killed.lock().unwrap().contains(&launch) {
                Poll::Dead
            } else {
                Poll::Alive
            },
        )
    }
}
