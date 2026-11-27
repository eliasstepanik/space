//! Trust on first use for the servers the client dials.
//!
//! A cluster with a secret presents its CA behind every certificate, and the
//! CA's public key is the cluster's **pin** (`space_net::ClusterPin`,
//! `docs/security-notes.md`). The first time this client dials a `host:port` it
//! trusts the pin the server presents and remembers it; every later dial of
//! that `host:port` must present the same one, or it is refused. Every dial the
//! client makes goes through [`join`]: the entry address and every shard a
//! redirect or a handoff names, each with its own entry.
//!
//! * **The store** is a JSON object, `{"host:port": "<64 hex characters>"}`, at
//!   `<config dir>/space/pins.json` — `$XDG_CONFIG_HOME` or `$HOME/.config` on
//!   Unix, `%APPDATA%` on Windows — or wherever the environment variable
//!   [`PIN_FILE_ENV`] (`SPACE_PIN_FILE`) says (tests set it). The key is the
//!   address as it was dialed (`ip:port`). A missing file is an empty store; an
//!   unreadable one is an empty store too, with a warning. A corrupt one is
//!   renamed to `pins.json.corrupt` (so nothing in it is lost), then is an
//!   empty store with a warning; the next learned pin writes a fresh file,
//!   through a temporary file and a rename. One process serializes its reads and
//!   writes of the file; two processes are not locked against each other.
//! * **A server without a cluster secret** presents no CA: there is no pin to
//!   learn, the dial goes ahead unpinned and nothing is stored. A server that
//!   *has* a stored pin and later presents none is refused.
//! * **`--pin HEX`** ([`set_pin`]) gives the pin to hold every server of this
//!   run to (one cluster has one pin, and a redirect leads to its other
//!   shards): TOFU is skipped, the store is neither read nor written, and a
//!   mismatch fails hard.
//! * **`--reset-pins`** ([`reset_pins`]) deletes the file.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use space_net::{client_endpoint_tofu, ClusterPin, Joined, NetError};
use space_protocol::Hello;

/// The environment variable that overrides where the pin store lives.
pub const PIN_FILE_ENV: &str = "SPACE_PIN_FILE";

/// Held while the store file is read or written.
static STORE_LOCK: Mutex<()> = Mutex::new(());

/// The pin `--pin` gave, if any.
static PINNED: Mutex<Option<ClusterPin>> = Mutex::new(None);

/// Hold every server this process dials to `pin` (`--pin`), instead of the
/// store; `None` goes back to the store.
pub fn set_pin(pin: Option<ClusterPin>) {
    *PINNED.lock().unwrap_or_else(PoisonError::into_inner) = pin;
}

fn pinned() -> Option<ClusterPin> {
    *PINNED.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The known pins: `host:port` to the cluster pin it presented the first time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PinStore {
    pins: BTreeMap<String, ClusterPin>,
}

impl PinStore {
    /// The pin remembered for `key` (`host:port`).
    #[must_use]
    pub fn get(&self, key: &str) -> Option<ClusterPin> {
        self.pins.get(key).copied()
    }

    /// Remember `pin` for `key`, replacing what was there.
    pub fn insert(&mut self, key: &str, pin: ClusterPin) {
        self.pins.insert(key.to_owned(), pin);
    }

    /// How many servers have a pin.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pins.len()
    }

    /// Whether no server has a pin.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pins.is_empty()
    }

    /// The store as the JSON text of the file.
    #[must_use]
    pub fn to_json(&self) -> String {
        let map: BTreeMap<&str, String> = self
            .pins
            .iter()
            .map(|(key, pin)| (key.as_str(), pin.to_string()))
            .collect();
        serde_json::to_string_pretty(&map).expect("a map of strings is JSON")
    }

    /// Read the JSON text of a file.
    ///
    /// # Errors
    ///
    /// Why the text is not an object of `host:port` to 64-hex-character pins.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let map: BTreeMap<String, String> =
            serde_json::from_str(text).map_err(|e| e.to_string())?;
        let mut pins = BTreeMap::new();
        for (key, pin) in map {
            let parsed = pin.parse().map_err(|e| format!("{key}: {e}"))?;
            pins.insert(key, parsed);
        }
        Ok(Self { pins })
    }

    /// Load the file at `path`: a missing file is an empty store, an unreadable
    /// one an empty store and a warning. A corrupt one is the same, and is
    /// moved aside to `<path>.corrupt` first, so that the fresh file the next
    /// learned pin writes does not destroy what was in it.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_json(&text).unwrap_or_else(|why| {
                let aside = sibling(path, ".corrupt");
                let moved = match std::fs::rename(path, &aside) {
                    Ok(()) => format!("moved to {}", aside.display()),
                    Err(e) => format!("could not be moved aside: {e}"),
                };
                log::warn!(
                    "space-client: the pin file {} is corrupt ({why}); treating it as empty \
                     ({moved})",
                    path.display()
                );
                Self::default()
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Self::default(),
            Err(e) => {
                log::warn!(
                    "space-client: cannot read the pin file {} ({e}); treating it as empty",
                    path.display()
                );
                Self::default()
            }
        }
    }

    /// Write the store to `path`, creating its directory. The text goes to a
    /// temporary file next to it, which is then renamed over `path`: a crash
    /// leaves the old file or the new one, never half of one.
    ///
    /// # Errors
    ///
    /// The I/O error of the directory or the file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let temporary = sibling(path, ".tmp");
        std::fs::write(&temporary, self.to_json())?;
        std::fs::rename(&temporary, path).inspect_err(|_| {
            let _ = std::fs::remove_file(&temporary);
        })
    }
}

/// `path` with `suffix` appended to its file name.
fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().map(OsString::from).unwrap_or_default();
    name.push(suffix);
    path.with_file_name(name)
}

/// The user-config directory from the environment's values.
fn config_dir(
    windows: bool,
    appdata: Option<OsString>,
    xdg: Option<OsString>,
    home: Option<OsString>,
) -> Option<PathBuf> {
    let non_empty = |v: Option<OsString>| v.filter(|v| !v.is_empty()).map(PathBuf::from);
    if windows {
        return non_empty(appdata);
    }
    non_empty(xdg).or_else(|| non_empty(home).map(|home| home.join(".config")))
}

/// Where the pin store is: [`PIN_FILE_ENV`], else `<config dir>/space/pins.json`;
/// `None` when the environment names no home to put it in.
#[must_use]
pub fn pin_file_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(PIN_FILE_ENV).filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(path));
    }
    config_dir(
        cfg!(windows),
        std::env::var_os("APPDATA"),
        std::env::var_os("XDG_CONFIG_HOME"),
        std::env::var_os("HOME"),
    )
    .map(|dir| dir.join("space").join("pins.json"))
}

/// Delete the pin file (`--reset-pins`). Returns where it is and whether there
/// was one.
///
/// # Errors
///
/// No place for the file is known, or it could not be deleted.
pub fn reset_pins() -> Result<(PathBuf, bool), String> {
    let path = pin_file_path().ok_or_else(|| {
        format!(
            "no place for the pin file: set {PIN_FILE_ENV}, or HOME / XDG_CONFIG_HOME / APPDATA"
        )
    })?;
    let _guard = STORE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok((path, true)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok((path, false)),
        Err(e) => Err(format!("cannot delete {}: {e}", path.display())),
    }
}

/// The pin stored for `key`, under the lock.
fn stored_pin(key: &str) -> Option<ClusterPin> {
    let path = pin_file_path()?;
    let _guard = STORE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    PinStore::load(&path).get(key)
}

/// Remember `pin` for `key` unless something is stored already.
fn remember(key: &str, pin: ClusterPin) {
    let Some(path) = pin_file_path() else {
        log::warn!("space-client: no place for the pin file; {key} pin {pin} is not stored");
        return;
    };
    let _guard = STORE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let mut store = PinStore::load(&path);
    if store.get(key).is_some() {
        return;
    }
    log::info!("space-client: trusting {key} pin {pin} for the first time; verify it out of band");
    store.insert(key, pin);
    if let Err(e) = store.save(&path) {
        log::warn!(
            "space-client: cannot write the pin file {}: {e}",
            path.display()
        );
    }
}

/// The refusal of a server that did not present the pin it is held to.
fn mismatch(
    key: &str,
    expected: ClusterPin,
    received: Option<ClusterPin>,
    from_flag: bool,
) -> NetError {
    let received = received.map_or_else(
        || "none (the server presented no cluster CA)".to_owned(),
        |pin| pin.to_string(),
    );
    let source = if from_flag {
        "given with --pin"
    } else {
        "stored from the first connection"
    };
    NetError::Untrusted(format!(
        "the certificate of {key} does not match its pin: expected {expected} ({source}), \
         received {received}. Someone may be impersonating this server, or its cluster secret \
         changed; if you know it did, run `space-client --reset-pins` to forget the stored \
         pins (or pass the new one with --pin)"
    ))
}

/// Say hello to `addr` as a player over a connection whose certificate is held
/// to this module's rules (see the [module docs](self)).
///
/// # Errors
///
/// As [`space_net::join`]; a certificate that is not the pinned cluster's is
/// [`NetError::Untrusted`] with a message that names `addr`, both pins and
/// `--reset-pins`.
pub async fn join(addr: SocketAddr, server_name: &str, hello: &Hello) -> Result<Joined, NetError> {
    let key = addr.to_string();
    let from_flag = pinned();
    let expected = from_flag.or_else(|| stored_pin(&key));
    let (endpoint, probe) =
        client_endpoint_tofu(addr, expected).map_err(|e| NetError::Handshake(e.to_string()))?;
    match space_net::join(&endpoint, addr, server_name, hello).await {
        Ok(joined) => {
            if expected.is_none() {
                match probe.presented_pin() {
                    Some(pin) => remember(&key, pin),
                    None => log::info!(
                        "space-client: {key} presents no cluster CA; connected without a pin"
                    ),
                }
            }
            Ok(joined)
        }
        Err(NetError::Untrusted(why)) => Err(match expected {
            Some(expected) if probe.presented_pin() == Some(expected) => {
                NetError::Untrusted(format!(
                    "{key} presents the pinned cluster CA ({expected}) but a certificate it did \
                     not issue ({why}). Someone may be impersonating this server; if the \
                     server's certificate setup really changed, run `space-client --reset-pins` \
                     (or pass the pin with --pin)"
                ))
            }
            Some(expected) => mismatch(&key, expected, probe.presented_pin(), from_flag.is_some()),
            None => NetError::Untrusted(why),
        }),
        Err(other) => Err(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin(byte: u8) -> ClusterPin {
        ClusterPin::from_bytes([byte; 32])
    }

    #[test]
    fn the_store_round_trips_through_json() {
        let mut store = PinStore::default();
        store.insert("127.0.0.1:7777", pin(0xab));
        store.insert("[::1]:7870", pin(0x01));
        let json = store.to_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["127.0.0.1:7777"], "ab".repeat(32));
        assert_eq!(PinStore::from_json(&json), Ok(store));
        assert_eq!(PinStore::from_json("{}"), Ok(PinStore::default()));
    }

    #[test]
    fn a_file_that_is_not_a_map_of_pins_is_an_error() {
        for bad in ["", "[]", "{\"a\": 1}", "{\"a:1\": \"abc\"}", "not json"] {
            assert!(PinStore::from_json(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_store_is_saved_with_its_directory_and_loaded_again() {
        let dir = std::env::temp_dir().join(format!("space-pins-unit-{}", std::process::id()));
        let path = dir.join("nested").join("pins.json");
        let mut store = PinStore::default();
        store.insert("1.2.3.4:5", pin(7));
        store.save(&path).unwrap();
        assert_eq!(PinStore::load(&path), store);
        std::fs::write(&path, "{ corrupt").unwrap();
        assert!(PinStore::load(&path).is_empty());
        assert!(PinStore::load(&dir.join("missing.json")).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_config_directory_follows_the_platform() {
        let os = |s: &str| Some(OsString::from(s));
        assert_eq!(
            config_dir(false, None, os("/x"), os("/home/u")),
            Some(PathBuf::from("/x"))
        );
        assert_eq!(
            config_dir(false, None, os(""), os("/home/u")),
            Some(PathBuf::from("/home/u/.config"))
        );
        assert_eq!(config_dir(false, None, None, None), None);
        assert_eq!(
            config_dir(true, os("C:\\Users\\u\\AppData\\Roaming"), os("/x"), None),
            Some(PathBuf::from("C:\\Users\\u\\AppData\\Roaming"))
        );
    }
}
