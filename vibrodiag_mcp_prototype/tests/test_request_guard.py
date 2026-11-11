"""Host and browser-origin guards for the web UI and the loopback model servers."""

import http.client
import importlib
import importlib.util
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from vibroagent_mcp import request_guard, webchat_server

# (module, handler class, attributes the server object needs, a path that answers 200 or None)
MODEL_SERVERS = [
    ("vibroagent_mcp.geniex_openai_server", "GenieXOpenAIHandler", {"geniex_runtime": SimpleNamespace()}, "/livez"),
    ("vibroagent_mcp.genie_openai_server", "GenieOpenAIHandler", {}, None),
    ("vibroagent_mcp.onnx_openai_server", "OnnxOpenAIHandler", {}, None),
    ("vibroagent_mcp.vibrogemma_geniex_server", "Handler", {"runtime": SimpleNamespace()}, "/v1/models"),
]


@pytest.fixture(autouse=True)
def _fixed_local_names(monkeypatch):
    monkeypatch.delenv("VIBRO_ALLOWED_WEB_HOSTS", raising=False)
    monkeypatch.setattr(request_guard.socket, "gethostname", lambda: "iq9075")


def _serve(handler_class, **server_attributes):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    for name, value in server_attributes.items():
        setattr(server, name, value)
    threading.Thread(target=server.serve_forever, name="guard-test", daemon=True).start()
    return server


def _request(server, method, path, *, host=None, headers=None, body=None):
    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host or f"127.0.0.1:{port}")
        for name, value in (headers or {}).items():
            conn.putheader(name, value)
        if body is not None:
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_host_header_accepts_ip_literals_and_this_machines_names():
    allowed = request_guard.host_header_allowed
    for host in (
        None,
        "",
        "127.0.0.1:7860",
        "192.168.1.50:7860",
        "192.168.1.50",
        "[::1]:7860",
        "[fe80::1]",
        "localhost:7860",
        "LOCALHOST",
        "iq9075:7860",
        "iq9075.local:7860",
        "iq9075.local.",
    ):
        assert allowed(host), host
    for host in (
        "evil.example:7860",
        "evil.example",
        "192.168.1.50.evil.example:7860",
        "localhost.evil.example",
        "iq9075.evil.example:7860",
        "[::1",
        "[::1]x",
        "host:port",
        ":7860",
        "a b",
    ):
        assert not allowed(host), host


def test_extra_host_names_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("VIBRO_ALLOWED_WEB_HOSTS", "vibro.example.org, Proxy.Internal ")
    assert request_guard.host_header_allowed("vibro.example.org:443")
    assert request_guard.host_header_allowed("proxy.internal")
    assert not request_guard.host_header_allowed("evil.example")


def test_local_service_refuses_rebinding_and_browser_requests():
    refusal = request_guard.local_service_refusal
    assert refusal({}) is None
    assert refusal({"Host": "127.0.0.1:18181"}) is None
    assert refusal({"Host": "127.0.0.1:18181", "Sec-Fetch-Site": "none"}) is None
    assert refusal({"Host": "evil.example:18181"}) == "host_not_allowed"
    assert refusal({"Host": "127.0.0.1:18181", "Origin": "http://evil.example"}) == "browser_origin_not_allowed"
    assert refusal({"Host": "127.0.0.1:18181", "Origin": "null"}) == "browser_origin_not_allowed"
    assert refusal({"Host": "127.0.0.1:18181", "Sec-Fetch-Site": "cross-site"}) == "browser_origin_not_allowed"


def test_webchat_refuses_a_rebinding_host_on_every_route(monkeypatch):
    monkeypatch.delenv("VIBRO_WEB_TOKEN", raising=False)  # the Host check must not depend on the token
    server = _serve(webchat_server.WebchatHandler)
    try:
        status, body = _request(server, "GET", "/health")
        assert status == 200 and json.loads(body)["ok"] is True

        for method, path in (("GET", "/health"), ("GET", "/"), ("GET", "/api/board-readers"), ("POST", "/api/chat")):
            status, body = _request(server, method, path, host="evil.example:7860", body=b"{}" if method == "POST" else None)
            assert status == 421, (method, path)
            assert json.loads(body)["error"] == "host_not_allowed"

        # A rebinding page is same-origin with its own hostname, so Origin matches Host; the
        # Host check is what stops it.
        status, _ = _request(
            server, "GET", "/api/board-readers",
            host="evil.example:7860", headers={"Origin": "http://evil.example:7860", "Sec-Fetch-Site": "same-origin"},
        )
        assert status == 421

        status, _ = _request(server, "GET", "/health", host="iq9075.local:7860")
        assert status == 200
        monkeypatch.setenv("VIBRO_ALLOWED_WEB_HOSTS", "vibro.example.org")
        status, _ = _request(server, "GET", "/health", host="vibro.example.org")
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("module_name, class_name, server_attributes, ok_path", MODEL_SERVERS)
def test_model_servers_refuse_rebinding_and_browser_requests(module_name, class_name, server_attributes, ok_path):
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"{module_name} is not part of this deployment tree")
    handler_class = getattr(importlib.import_module(module_name), class_name)
    server = _serve(handler_class, **server_attributes)
    try:
        for method in ("GET", "POST"):
            body = b"{}" if method == "POST" else None
            path = "/v1/chat/completions" if method == "POST" else "/v1/models"
            status, payload = _request(server, method, path, host="evil.example:18181", body=body)
            assert status == 403, (method, "host")
            assert json.loads(payload)["error"]["message"] == "host_not_allowed"

            status, payload = _request(server, method, path, headers={"Origin": "http://evil.example"}, body=body)
            assert status == 403, (method, "origin")
            assert json.loads(payload)["error"]["message"] == "browser_origin_not_allowed"

            status, _ = _request(server, method, path, headers={"Sec-Fetch-Site": "cross-site"}, body=body)
            assert status == 403, (method, "sec-fetch-site")

        if ok_path is not None:
            # A local Python client (no Origin, loopback Host) is still served.
            status, _ = _request(server, "GET", ok_path)
            assert status == 200
    finally:
        server.shutdown()
        server.server_close()
