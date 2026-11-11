"""Request-origin guards shared by the web UI and the loopback model servers.

Standard library only: the model servers import this from the GenieX environment, which
does not carry the web application's dependencies.

Two attacks are closed here. In DNS rebinding a web page served from ``evil.example`` re-points
that name at this host, so the browser treats the page and this service as one origin; the
request then arrives with ``Host: evil.example``. In localhost cross-site requests a page in a
browser on this machine posts straight to ``http://127.0.0.1:<port>``; such a request always
carries an ``Origin`` header. Neither matters to the real callers: browsers that opened the UI by
address, and local Python clients that never send ``Origin``.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any, Mapping

ALLOWED_HOSTS_ENV = "VIBRO_ALLOWED_WEB_HOSTS"


def allowed_host_names() -> set[str]:
    """Host names a request may address this machine by, besides IP literals.

    ``localhost``, this machine's own hostname (also as ``<hostname>.local`` for mDNS), and the
    comma-separated names in ``VIBRO_ALLOWED_WEB_HOSTS`` (for example a reverse-proxy or VPN name).
    """
    names = {"localhost"}
    try:
        local = socket.gethostname().strip().lower().rstrip(".")
    except OSError:
        local = ""
    if local:
        names.add(local)
        names.add(f"{local}.local")
    for item in os.environ.get(ALLOWED_HOSTS_ENV, "").split(","):
        item = item.strip().lower().rstrip(".")
        if item:
            names.add(item)
    return names


def host_header_name(host_header: str) -> str | None:
    """Return the host of a ``Host`` header without its port, or None when it is malformed."""
    host = host_header.strip().lower()
    if not host:
        return None
    if host.startswith("["):
        end = host.find("]")
        if end < 0:
            return None
        rest = host[end + 1 :]
        if rest and not (rest.startswith(":") and rest[1:].isdigit()):
            return None
        return host[1:end]
    name, separator, port = host.rpartition(":")
    if separator:
        if not name or not port.isdigit():
            return None
        host = name
    return host.rstrip(".")


def host_header_allowed(host_header: str | None) -> bool:
    """True when the ``Host`` header names this machine by IP literal or by an allowed name.

    A missing header is accepted: only non-browser HTTP/1.0 clients omit it, and rebinding needs a
    browser.
    """
    if host_header is None or not host_header.strip():
        return True
    name = host_header_name(host_header)
    if name is None:
        return False
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return name in allowed_host_names()
    return True


def local_service_refusal(headers: Mapping[str, Any]) -> str | None:
    """Why a local model server must refuse a request, or None when it may be served.

    Model servers are called by local Python clients. A request that names another host, or that
    a browser marked as coming from a web page, is refused before any model work starts.
    """
    if not host_header_allowed(headers.get("Host")):
        return "host_not_allowed"
    if str(headers.get("Origin") or "").strip():
        return "browser_origin_not_allowed"
    if str(headers.get("Sec-Fetch-Site") or "").strip().lower() in {"cross-site", "same-site"}:
        return "browser_origin_not_allowed"
    return None
