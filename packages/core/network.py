"""
Libra Core - Network Utilities and Windows DNS Optimization
"""

from __future__ import annotations

import logging
import socket
from typing import Any

logger = logging.getLogger("libra.network")
_is_patched = False


def enable_ipv4_preference() -> None:
    """
    On Windows environments with inactive or misconfigured IPv6 adapters
    (e.g., Tailscale, WSL, dead fec0:: site-local anycast DNS servers),
    getaddrinfo(family=0) queries IPv6 DNS servers first and hangs for 20-40s
    before falling back to IPv4, causing httpx timeouts and connection errors.

    This function configures socket.getaddrinfo to resolve standard domain names
    via AF_INET (IPv4) unless IPv6 is specifically requested or the host is a local literal.
    """
    global _is_patched
    if _is_patched:
        return

    orig_getaddrinfo = socket.getaddrinfo

    def _fast_ipv4_getaddrinfo(
        host: Any,
        port: Any,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[Any]:
        if family == 0:
            if isinstance(host, str) and (host in ("localhost", "::1") or ":" in host):
                return orig_getaddrinfo(host, port, family, type, proto, flags)
            family = socket.AF_INET
        return orig_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = _fast_ipv4_getaddrinfo
    _is_patched = True
    logger.debug("Fast IPv4 DNS resolution preference enabled.")


enable_ipv4_preference()
