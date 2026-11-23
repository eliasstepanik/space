//! The process launcher: one `space-server` process per shard.
//!
//! [`ProcessLauncher`] starts the `space-server` executable **lying next to the
//! orchestrator's own executable** (or the one it is told to use), once per
//! [`SpawnSpec`]:
//!
//! ```text
//!   space-server --orchestrator <control address> --bind <ip>:<host port> --cluster <id>
//! ```
//!
//! No `--shard`, no region: the shard registers without either and is given
//! both at its hello, like a shard started by hand. What makes a launched
//! shard different is only who started it — and so who watches it:
//!
//! * **Executable.** The path is made absolute once, when the launcher is
//!   built, and exactly that file is started: a bare name is never looked up
//!   in `PATH`, and a relative one does not follow a change of directory.
//! * **Address.** It listens on the spec's host port (the orchestrator hands
//!   out one port per shard from its configured range), on the IP the control
//!   port is bound to, or loopback when that is the wildcard. That address is
//!   what its hello advertises, which is how the orchestrator knows which shard
//!   this launch became ([`ShardHandle::endpoint`]).
//! * **Secret.** The cluster secret is handed over in the child's environment
//!   (`SPACE_CLUSTER_SECRET`), never on its command line.
//! * **Environment.** The child inherits the orchestrator's environment
//!   **without the orchestrator's own settings**: every variable whose name
//!   starts with [`WITHHELD_ENV_PREFIX`] — the web UI's admin token among
//!   them — and those of [`WITHHELD_ENV`] are taken out. A shard faces
//!   players; it has no business holding the operator's credentials.
//! * **Log.** Its standard output and standard error go to
//!   `shard-<host port>.log` in the log directory, appended (a replacement that
//!   gets the port of a shard before it continues the same file).
//! * **Pid.** [`PID_FILE`] in the log directory lists the pid of every child
//!   that is running — exactly those: it is written anew from the launcher's
//!   own table whenever a child is started, stopped or seen to have exited,
//!   and removed when none is left. A script that stops the orchestrator reads
//!   it to make sure nothing it started stays behind.
//! * **Liveness.** [`Launcher::poll`] asks the operating system whether the
//!   child has exited; [`Launcher::kill`] stops it and reaps it.
//!
//! If the orchestrator itself is killed its children are not (nothing runs to
//! stop them, and the pid file keeps listing them) — they stop by themselves
//! once they notice the orchestrator is gone (`space_server::orchestrated`: a
//! shard does not run without one), and a launcher that is dropped normally
//! stops what it started.
//!
//! Built on [`std::process`]: the workspace's tokio has no `process` feature,
//! and none of this needs one — starting, asking and stopping a child do not
//! block.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, MutexGuard, PoisonError};

use space_server::orchestrated::{ENV_CLUSTER_SECRET, ENV_CLUSTER_SECRET_FILE};

use super::{dialable, LaunchError, LaunchId, Launcher, Poll, ShardHandle, SpawnSpec};

/// The file, in the log directory, that lists the pid of every shard process
/// of this launcher that is running (one per line; see the module docs).
pub const PID_FILE: &str = "launched-shards.pid";

/// The name of the shard executable next to the orchestrator's.
pub const SHARD_EXE: &str = if cfg!(windows) {
    "space-server.exe"
} else {
    "space-server"
};

/// A child does not inherit any variable whose name starts with this: they
/// are the orchestrator's own settings (`SPACE_ORCHESTRATOR_WEBUI_TOKEN` is
/// the web UI's admin password).
pub const WITHHELD_ENV_PREFIX: &str = "SPACE_ORCHESTRATOR_";

/// Further variables a child does not inherit: credentials of other parts of
/// the stack that may be set where the orchestrator is started.
pub const WITHHELD_ENV: &[&str] = &["SPACE_HARNESS_TOKEN"];

/// Whether the variable `name` is kept from a child ([`WITHHELD_ENV_PREFIX`],
/// [`WITHHELD_ENV`]). Compared without regard to case: on Windows a variable's
/// name has none.
fn withheld(name: &OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    let name = name.to_ascii_uppercase();
    name.starts_with(WITHHELD_ENV_PREFIX) || WITHHELD_ENV.contains(&name.as_str())
}

/// One child process.
#[derive(Debug)]
struct Shard {
    child: Child,
    endpoint: SocketAddr,
    /// It was seen to have exited ([`Launcher::poll`]). Its pid is no longer
    /// its own — the system may give it to anybody — so it is no longer in
    /// the pid file.
    exited: bool,
}

/// Starts one `space-server` process per shard (see the module docs).
#[derive(Debug)]
pub struct ProcessLauncher {
    /// The executable to start: an absolute path.
    exe: PathBuf,
    /// Where the per-shard logs and the pid file go.
    log_dir: PathBuf,
    /// The IP a launched shard binds (see [`ProcessLauncher::new`]).
    bind_ip: IpAddr,
    /// What is running. Never held across an await (there are none here).
    children: Mutex<HashMap<LaunchId, Shard>>,
}

impl ProcessLauncher {
    /// A launcher that starts `exe` and logs below `log_dir`. Launched shards
    /// listen on the IP of `control_bind` — the orchestrator's own host — or
    /// on loopback when that is the wildcard address.
    ///
    /// A relative `exe` is taken relative to the working directory **now**
    /// and kept as an absolute path: [`Command`] looks a bare name up in
    /// `PATH`, so `--shard-exe space-server` would otherwise be checked in
    /// the working directory ([`Self::ensure_available`]) and then started
    /// from wherever `PATH` finds one.
    #[must_use]
    pub fn new(
        exe: impl Into<PathBuf>,
        log_dir: impl Into<PathBuf>,
        control_bind: SocketAddr,
    ) -> Self {
        let bind_ip = if control_bind.ip().is_unspecified() {
            match control_bind {
                SocketAddr::V4(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
                SocketAddr::V6(_) => IpAddr::V6(Ipv6Addr::LOCALHOST),
            }
        } else {
            control_bind.ip()
        };
        let exe = exe.into();
        // (Only an empty path cannot be made absolute; it is then left as it
        // is and `ensure_available` says that it does not exist.)
        let exe = std::path::absolute(&exe).unwrap_or(exe);
        Self {
            exe,
            log_dir: log_dir.into(),
            bind_ip,
            children: Mutex::new(HashMap::new()),
        }
    }

    /// The `space-server` executable lying next to the running executable.
    ///
    /// # Errors
    ///
    /// The running executable's path cannot be determined.
    pub fn default_exe() -> Result<PathBuf, LaunchError> {
        let own = std::env::current_exe().map_err(|e| {
            LaunchError::Unavailable(format!(
                "cannot tell where this executable lies ({e}), so not where {SHARD_EXE} does; \
                 name it with --shard-exe"
            ))
        })?;
        Ok(own.with_file_name(SHARD_EXE))
    }

    /// Whether this launcher can start anything: the executable exists and the
    /// log directory can be created. Lets the binary fail at startup, with the
    /// path, instead of at the first launch.
    ///
    /// # Errors
    ///
    /// [`LaunchError::Unavailable`] naming what is missing.
    pub fn ensure_available(&self) -> Result<(), LaunchError> {
        if !self.exe.is_file() {
            return Err(LaunchError::Unavailable(format!(
                "the shard executable {} does not exist (the process launcher starts the \
                 {SHARD_EXE} lying next to the orchestrator's own executable; name another \
                 with --shard-exe)",
                self.exe.display()
            )));
        }
        std::fs::create_dir_all(&self.log_dir).map_err(|e| {
            LaunchError::Unavailable(format!(
                "the shard log directory {} cannot be created: {e}",
                self.log_dir.display()
            ))
        })
    }

    /// The executable this launcher starts (an absolute path).
    #[must_use]
    pub fn exe(&self) -> &Path {
        &self.exe
    }

    /// The log file of the shard on `host_port`.
    #[must_use]
    pub fn log_path(&self, host_port: u16) -> PathBuf {
        self.log_dir.join(format!("shard-{host_port}.log"))
    }

    /// The pid file (see [`PID_FILE`]).
    #[must_use]
    pub fn pid_path(&self) -> PathBuf {
        self.log_dir.join(PID_FILE)
    }

    /// The operating-system pid of what `launch` started, while it is
    /// supervised.
    #[must_use]
    pub fn pid_of(&self, launch: LaunchId) -> Option<u32> {
        self.table().get(&launch).map(|shard| shard.child.id())
    }

    /// The pid of the child that listens at `endpoint` and is running as far
    /// as this launcher knows (a shard is known to everybody else by its
    /// address, not by its launch).
    #[must_use]
    pub fn pid_at(&self, endpoint: SocketAddr) -> Option<u32> {
        self.table()
            .values()
            .find(|shard| shard.endpoint == endpoint && !shard.exited)
            .map(|shard| shard.child.id())
    }

    /// The pids of the children that are running as far as this launcher
    /// knows — started, not stopped, not seen to have exited — in the order
    /// they were launched. This is what the pid file lists.
    #[must_use]
    pub fn live_pids(&self) -> Vec<u32> {
        Self::live(&self.table())
    }

    /// The table of children. A poisoned lock is taken all the same: the
    /// table holds processes that still have to be stopped, and nothing done
    /// under the lock leaves it half-changed.
    fn table(&self) -> MutexGuard<'_, HashMap<LaunchId, Shard>> {
        self.children.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The pids [`Self::live_pids`] returns, from the table itself.
    fn live(children: &HashMap<LaunchId, Shard>) -> Vec<u32> {
        let mut live: Vec<(LaunchId, u32)> = children
            .iter()
            .filter(|(_, shard)| !shard.exited)
            .map(|(launch, shard)| (*launch, shard.child.id()))
            .collect();
        live.sort_unstable();
        live.into_iter().map(|(_, pid)| pid).collect()
    }

    /// Write the pid file anew from `children`: the pid of every child that
    /// is running, one per line — or no file when there is none.
    ///
    /// Called with the table's lock held, after every change to it, so the
    /// file never names a process this launcher has stopped or seen exit (a
    /// pid that is free again may be anybody's next). Written to a file
    /// beside it first and moved into place: a reader sees the old list or
    /// the new one, never half a line.
    ///
    /// A failure is reported, not fatal: the shards run, the list a stop
    /// script reads is only out of date (and they stop by themselves when
    /// the orchestrator is gone).
    fn write_pid_file(&self, children: &HashMap<LaunchId, Shard>) {
        let path = self.pid_path();
        let live = Self::live(children);
        let written = if live.is_empty() {
            match std::fs::remove_file(&path) {
                Err(err) if err.kind() != std::io::ErrorKind::NotFound => Err(err),
                _ => Ok(()),
            }
        } else {
            let text: String = live.iter().map(|pid| format!("{pid}\n")).collect();
            let beside = self.log_dir.join(format!("{PID_FILE}.tmp"));
            std::fs::write(&beside, text).and_then(|()| std::fs::rename(&beside, &path))
        };
        if let Err(err) = written {
            eprintln!(
                "space-orchestrator: could not bring {} up to date: {err}",
                path.display()
            );
        }
    }

    /// The command that starts the shard of `spec` on `endpoint`: the
    /// executable, its arguments and its environment (the standard streams
    /// are [`Self::start`]'s). `inherited` names the variables the child
    /// would inherit — the orchestrator's own environment.
    fn command(
        &self,
        spec: &SpawnSpec,
        endpoint: SocketAddr,
        inherited: impl IntoIterator<Item = OsString>,
    ) -> Command {
        // `self.exe` is absolute: exactly that file, never a `PATH` lookup.
        let mut command = Command::new(&self.exe);
        command
            .arg("--orchestrator")
            .arg(dialable(spec.control_addr).to_string())
            .arg("--bind")
            .arg(endpoint.to_string())
            .arg("--cluster")
            .arg(&spec.cluster);
        // The orchestrator's own settings stay with the orchestrator — its
        // web UI token above all. (Not `env_clear`: a process needs the
        // system's variables, on Windows to start at all.)
        for name in inherited {
            if withheld(&name) {
                command.env_remove(name);
            }
        }
        // The secret travels in the environment, never in argv (which every
        // process list shows). An orchestrator that has none must not pass on
        // one of its own environment by accident either.
        if spec.secret.is_empty() {
            command.env_remove(ENV_CLUSTER_SECRET);
        } else {
            command.env(ENV_CLUSTER_SECRET, &spec.secret);
        }
        command.env_remove(ENV_CLUSTER_SECRET_FILE);
        command
    }

    /// Start the child for `spec`.
    fn start(&self, spec: &SpawnSpec) -> Result<Shard, LaunchError> {
        let failed = |what: &str, err: std::io::Error| LaunchError::Spawn(format!("{what}: {err}"));
        if spec.host_port == 0 {
            return Err(LaunchError::Spawn(
                "the process launcher needs a host port for the shard (none of the configured \
                 range is free)"
                    .into(),
            ));
        }
        std::fs::create_dir_all(&self.log_dir)
            .map_err(|e| failed(&format!("creating {}", self.log_dir.display()), e))?;
        let log_path = self.log_path(spec.host_port);
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| failed(&format!("opening {}", log_path.display()), e))?;
        let log_err = log
            .try_clone()
            .map_err(|e| failed(&format!("opening {}", log_path.display()), e))?;

        let endpoint = SocketAddr::new(self.bind_ip, spec.host_port);
        let inherited = std::env::vars_os().map(|(name, _)| name);
        let child = self
            .command(spec, endpoint, inherited)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err))
            .spawn()
            .map_err(|e| failed(&format!("starting {}", self.exe.display()), e))?;
        Ok(Shard {
            child,
            endpoint,
            exited: false,
        })
    }
}

impl Drop for ProcessLauncher {
    /// Stop what is still running: an orchestrator that ends normally leaves
    /// no shard behind, and no pid file. (One that is killed cannot do this;
    /// its shards stop by themselves when they notice.)
    fn drop(&mut self) {
        // A poisoned lock must not leave the children running.
        let children = std::mem::take(
            self.children
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner),
        );
        for (_, mut shard) in children {
            let _ = shard.child.kill();
            let _ = shard.child.wait();
        }
        self.write_pid_file(&HashMap::new());
    }
}

impl Launcher for ProcessLauncher {
    async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
        let shard = self.start(&spec)?;
        let handle = ShardHandle {
            launch: spec.launch,
            // Not known here: the child registers by itself. The orchestrator
            // finds it by the address it advertises.
            shard: None,
            endpoint: shard.endpoint,
            process: format!("pid {}", shard.child.id()),
        };
        let mut children = self.table();
        if let Some(mut old) = children.insert(spec.launch, shard) {
            // A launch id is used once; should one ever come twice, the first
            // child must not be left unsupervised.
            let _ = old.child.kill();
            let _ = old.child.wait();
        }
        self.write_pid_file(&children);
        Ok(handle)
    }

    async fn kill(&self, launch: LaunchId) -> Result<(), LaunchError> {
        let mut children = self.table();
        // Not running: killing it is a no-op, not an error.
        let Some(mut shard) = children.remove(&launch) else {
            return Ok(());
        };
        // `kill` on a child that has already exited is fine; `wait` reaps it
        // and returns at once after a kill. Only then is it taken off the
        // list: a child that is still running is never missing from it.
        let _ = shard.child.kill();
        let reaped = shard.child.wait();
        self.write_pid_file(&children);
        reaped
            .map(|_| ())
            .map_err(|e| LaunchError::Spawn(format!("reaping the shard of {launch}: {e}")))
    }

    async fn poll(&self, launch: LaunchId) -> Result<Poll, LaunchError> {
        let mut children = self.table();
        let Some(shard) = children.get_mut(&launch) else {
            return Ok(Poll::Dead);
        };
        if shard.exited {
            return Ok(Poll::Dead);
        }
        match shard.child.try_wait() {
            Ok(None) => Ok(Poll::Alive),
            Ok(Some(_exited)) => {
                // Reaped: its pid is free for the system to hand out again,
                // so it comes off the list now, not when it is `kill`ed.
                shard.exited = true;
                self.write_pid_file(&children);
                Ok(Poll::Dead)
            }
            Err(e) => Err(LaunchError::Spawn(format!(
                "asking after the shard of {launch}: {e}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "space-orch-process-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ))
    }

    fn spec(launch: u64, host_port: u16) -> SpawnSpec {
        SpawnSpec {
            launch: LaunchId(launch),
            cluster: "test".into(),
            control_addr: "127.0.0.1:7900".parse().unwrap(),
            host_port,
            secret: String::new(),
        }
    }

    /// The pids the pid file lists; none when there is no file.
    fn noted(launcher: &ProcessLauncher) -> Vec<u32> {
        match std::fs::read_to_string(launcher.pid_path()) {
            Ok(text) => text
                .lines()
                .map(|line| line.trim().parse().expect("a pid a line"))
                .collect(),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(err) => panic!("the pid file cannot be read: {err}"),
        }
    }

    #[test]
    fn the_shard_executable_lies_next_to_the_running_one() {
        let exe = ProcessLauncher::default_exe().expect("this test has a path");
        assert_eq!(
            exe.parent(),
            std::env::current_exe().expect("own path").parent()
        );
        assert_eq!(exe.file_name().and_then(|n| n.to_str()), Some(SHARD_EXE));
    }

    #[test]
    fn launched_shards_bind_the_control_ports_ip_or_loopback() {
        let on = |bind: &str| ProcessLauncher::new("x", "y", bind.parse().unwrap()).bind_ip;
        assert_eq!(on("127.0.0.1:7900"), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(on("0.0.0.0:7900"), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(on("[::]:7900"), IpAddr::V6(Ipv6Addr::LOCALHOST));
        assert_eq!(on("10.0.0.5:7900"), "10.0.0.5".parse::<IpAddr>().unwrap());
    }

    /// `--shard-exe` may be a bare name or a relative path. It is checked as
    /// a file in the working directory — and that file is what must be
    /// started, not whatever `PATH` holds under the same name.
    ///
    /// Mutation this catches: keeping the path as given (`Command::new` of a
    /// bare name searches `PATH`: the check and the start would look in two
    /// different places).
    #[test]
    fn the_executable_is_an_absolute_path_and_exactly_that_is_started() {
        let cwd = std::env::current_dir().expect("a working directory");
        let bind: SocketAddr = "127.0.0.1:7900".parse().unwrap();
        for given in ["space-server", "bin/space-server", "./space-server"] {
            let launcher = ProcessLauncher::new(given, "logs", bind);
            assert!(launcher.exe().is_absolute(), "{given}");
            assert!(launcher.exe().starts_with(&cwd), "{given}");
            let command = launcher.command(&spec(1, 7_870), "127.0.0.1:7870".parse().unwrap(), []);
            assert_eq!(
                Path::new(command.get_program()),
                launcher.exe(),
                "{given}: the program started is the absolute path"
            );
            assert!(Path::new(command.get_program()).is_absolute(), "{given}");
        }
        // An absolute path is kept as it is.
        let own = std::env::current_exe().expect("own path");
        assert_eq!(ProcessLauncher::new(&own, "logs", bind).exe(), own);
    }

    /// What a child is started with: the three arguments, the cluster secret
    /// in its environment — and **not** the orchestrator's own settings.
    ///
    /// Mutation this catches: letting the child inherit everything (the web
    /// UI's admin token would sit in the environment of every player-facing
    /// shard process), or clearing everything (a Windows process does not
    /// start without the system's variables).
    #[test]
    fn a_child_is_not_handed_the_orchestrators_own_settings() {
        let launcher = ProcessLauncher::new("x", "y", "127.0.0.1:7900".parse().unwrap());
        let inherited = [
            "PATH",
            "SystemRoot",
            "SPACE_CLUSTER",
            "SPACE_ORCHESTRATOR_WEBUI",
            "SPACE_ORCHESTRATOR_WEBUI_TOKEN",
            "SPACE_ORCHESTRATOR_WEBUI_TOKEN_FILE",
            "SPACE_ORCHESTRATOR_PEER_TOKEN",
            "space_orchestrator_min_shards",
            "SPACE_HARNESS_TOKEN",
            "SPACE_CLUSTER_SECRET",
            "SPACE_CLUSTER_SECRET_FILE",
        ];
        let command_for = |secret: &str| {
            let mut spec = spec(3, 7_871);
            secret.clone_into(&mut spec.secret);
            launcher.command(
                &spec,
                "127.0.0.1:7871".parse().unwrap(),
                inherited.map(OsString::from),
            )
        };
        let command = command_for("handed-over-in-the-environment");
        let args: Vec<&OsStr> = command.get_args().collect();
        assert_eq!(
            args,
            [
                "--orchestrator",
                "127.0.0.1:7900",
                "--bind",
                "127.0.0.1:7871",
                "--cluster",
                "test"
            ]
        );
        // (The secret is never on the command line: it is not among these.)
        // What the command says about the environment: `None` = taken out,
        // `Some` = set, not mentioned = inherited as it is.
        let env: HashMap<String, Option<String>> = command
            .get_envs()
            .map(|(name, value)| {
                (
                    name.to_string_lossy().into_owned(),
                    value.map(|v| v.to_string_lossy().into_owned()),
                )
            })
            .collect();
        for gone in [
            "SPACE_ORCHESTRATOR_WEBUI",
            "SPACE_ORCHESTRATOR_WEBUI_TOKEN",
            "SPACE_ORCHESTRATOR_WEBUI_TOKEN_FILE",
            "SPACE_ORCHESTRATOR_PEER_TOKEN",
            "space_orchestrator_min_shards",
            "SPACE_HARNESS_TOKEN",
            "SPACE_CLUSTER_SECRET_FILE",
        ] {
            assert_eq!(env.get(gone), Some(&None), "{gone} must be taken out");
        }
        for kept in ["PATH", "SystemRoot", "SPACE_CLUSTER"] {
            assert_eq!(env.get(kept), None, "{kept} is inherited untouched");
        }
        assert_eq!(
            env.get("SPACE_CLUSTER_SECRET"),
            Some(&Some("handed-over-in-the-environment".to_owned()))
        );
        // No secret to hand over: the orchestrator's own is not passed on.
        let command = command_for("");
        let removed: Vec<_> = command
            .get_envs()
            .filter(|(name, value)| *name == "SPACE_CLUSTER_SECRET" && value.is_none())
            .collect();
        assert_eq!(removed.len(), 1);
    }

    /// A launcher whose executable is not there says so — at startup through
    /// `ensure_available`, and as a launch error rather than a panic.
    #[tokio::test]
    async fn a_missing_executable_is_an_error_naming_the_path() {
        let dir = temp_dir("missing");
        let exe = dir.join("no-such-space-server");
        let launcher = ProcessLauncher::new(&exe, &dir, "127.0.0.1:7900".parse().unwrap());
        let err = launcher.ensure_available().expect_err("nothing to start");
        assert!(
            matches!(&err, LaunchError::Unavailable(why) if why.contains("no-such-space-server")),
            "{err:?}"
        );
        // One sentence on one line: a string continuation that is written
        // `\\` + newline prints a backslash, a line break and the next line's
        // indentation.
        let said = err.to_string();
        assert!(!said.contains('\n'), "{said:?}");
        assert!(
            said.contains("starts the space-server") && said.contains("; name another with"),
            "{said:?}"
        );
        let err = launcher.spawn(spec(1, 7_870)).await.expect_err("no exe");
        assert!(matches!(err, LaunchError::Spawn(_)), "{err:?}");
        assert_eq!(launcher.poll(LaunchId(1)).await.unwrap(), Poll::Dead);
        launcher.kill(LaunchId(1)).await.expect("a no-op");
        // No port, no launch: the orchestrator ran out of its range.
        let err = launcher.spawn(spec(2, 0)).await.expect_err("no port");
        assert!(
            matches!(&err, LaunchError::Spawn(why) if why.contains("host port")),
            "{err:?}"
        );
        let said = err.to_string();
        assert!(
            !said.contains('\n') && said.contains("configured range is free"),
            "{said:?}"
        );
        assert!(
            noted(&launcher).is_empty(),
            "nothing started, nothing noted"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The mechanics, with a child that is not a shard at all: this test's own
    /// executable, asked to list nothing — it starts, is noted in the pid
    /// file, writes its log, and exits; `poll` then says so and takes it off
    /// the list, and `kill` reaps it without complaint.
    #[tokio::test]
    async fn a_child_is_started_logged_noted_and_seen_to_exit() {
        let dir = temp_dir("mechanics");
        let own = std::env::current_exe().expect("own path");
        let launcher = ProcessLauncher::new(&own, &dir, "127.0.0.1:7900".parse().unwrap());
        launcher.ensure_available().expect("this executable exists");
        // The test harness does not know `--orchestrator`: it prints an error
        // and exits. That is all this test needs of a child.
        let handle = launcher.spawn(spec(5, 7_871)).await.expect("starts");
        assert_eq!(handle.launch, LaunchId(5));
        assert_eq!(handle.shard, None, "the launcher cannot know the id");
        assert_eq!(handle.endpoint, "127.0.0.1:7871".parse().unwrap());
        let pid = launcher.pid_of(LaunchId(5)).expect("supervised");
        assert_eq!(handle.process, format!("pid {pid}"));
        assert_eq!(noted(&launcher), vec![pid]);
        assert_eq!(launcher.live_pids(), vec![pid]);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while launcher.poll(LaunchId(5)).await.expect("poll") == Poll::Alive {
            assert!(
                std::time::Instant::now() < deadline,
                "the child should have exited"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        // Seen to have exited: its pid may be anybody's now, so the list no
        // longer names it — and with nothing running there is no list.
        assert!(launcher.live_pids().is_empty());
        assert!(
            !launcher.pid_path().exists(),
            "an exited child is not in the pid file: {:?}",
            noted(&launcher)
        );
        let log = std::fs::read_to_string(launcher.log_path(7_871)).expect("its log");
        assert!(!log.is_empty(), "what the child said is in its log");
        launcher.kill(LaunchId(5)).await.expect("reaped");
        assert_eq!(launcher.pid_of(LaunchId(5)), None);
        assert_eq!(launcher.poll(LaunchId(5)).await.unwrap(), Poll::Dead);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The pid file is the launcher's table, written out: it lists every
    /// child that was started and neither stopped nor seen to exit, in launch
    /// order, and nothing else — a stopped child's pid goes at once, and the
    /// file goes with the last one. A poisoned lock changes none of that.
    ///
    /// Mutation this catches: appending pids (the state before: the pids of
    /// replaced and exited shards stayed in the file for ever, and a stop
    /// script went on to kill whatever had been given such a pid since).
    #[tokio::test]
    async fn the_pid_file_lists_exactly_the_children_that_are_running() {
        let dir = temp_dir("pids");
        let own = std::env::current_exe().expect("own path");
        let launcher = std::sync::Arc::new(ProcessLauncher::new(
            &own,
            &dir,
            "127.0.0.1:7900".parse().unwrap(),
        ));
        assert!(noted(&launcher).is_empty());
        // (The children exit by themselves at once — see the test above —
        // but nothing here asks, so as far as the launcher knows they run.)
        launcher.spawn(spec(1, 7_872)).await.expect("starts");
        launcher.spawn(spec(2, 7_873)).await.expect("starts");
        launcher.spawn(spec(3, 7_874)).await.expect("starts");
        let pid = |n: u64| launcher.pid_of(LaunchId(n)).expect("supervised");
        let (first, second, third) = (pid(1), pid(2), pid(3));
        assert_eq!(noted(&launcher), vec![first, second, third]);
        assert_eq!(noted(&launcher), launcher.live_pids());
        assert!(
            !dir.join(format!("{PID_FILE}.tmp")).exists(),
            "written beside and moved into place"
        );

        launcher.kill(LaunchId(2)).await.expect("stopped");
        assert_eq!(
            noted(&launcher),
            vec![first, third],
            "the stopped one is gone"
        );

        // The lock is poisoned (a thread panicked while it held it): the
        // launcher goes on, and so does the list.
        let poisoner = std::sync::Arc::clone(&launcher);
        let panicked = std::thread::spawn(move || {
            let _held = poisoner.children.lock().expect("not poisoned yet");
            panic!("poisoning the launcher's table on purpose");
        })
        .join();
        assert!(panicked.is_err() && launcher.children.is_poisoned());
        launcher.spawn(spec(4, 7_875)).await.expect("starts");
        let fourth = pid(4);
        assert_eq!(noted(&launcher), vec![first, third, fourth]);
        launcher.kill(LaunchId(1)).await.expect("stopped");
        assert_eq!(noted(&launcher), vec![third, fourth]);

        // Dropped — with the lock still poisoned: everything is stopped and
        // the file is gone.
        let pid_path = launcher.pid_path();
        let launcher = std::sync::Arc::try_unwrap(launcher).expect("the only handle");
        drop(launcher);
        assert!(!pid_path.exists(), "no child, no pid file");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
