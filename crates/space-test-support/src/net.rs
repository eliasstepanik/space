//! Addresses for tests.

use std::net::SocketAddr;

/// `127.0.0.1:0`: loopback, a port the system chooses.
pub fn loopback0() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_is_loopback_with_port_zero() {
        let addr = loopback0();
        assert!(addr.ip().is_loopback());
        assert_eq!(addr.port(), 0);
    }
}
