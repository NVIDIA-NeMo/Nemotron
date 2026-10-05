"""A real HTTPS oracle serving the BFCL endpoint contract, for probing over a socket.

The HTTP adapter is only worth certifying above A0 if the observations came from calls
that actually crossed a connection, so this fixture is a genuine TLS server rather than a
patched opener. It implements the endpoint contract directly instead of reusing any
production code, which is what lets a test tell the difference between a real regression
and a shared bug.

One session is one isolated episode. Fixtures arrive at session open, so two sessions
opened with the same fixtures see the same world, and a session that is deleted takes its
state with it.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import ipaddress
import json
import ssl
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import PROTOCOL_VERSION

ORACLE_ID = "library-endpoint"
ORACLE_VERSION = "1.0.0"


def write_localhost_certificate(
    root: Path,
    *,
    certificate_root: Path | None = None,
    host: str = "127.0.0.1",
) -> tuple[Path, Path]:
    """Issue a short-lived certificate a client can verify against ``host``.

    The certificate can be placed apart from the key, because a reviewed HTTP package has
    to contain the bundle it pins and has no business containing a private key. ``host`` is
    a parameter because a published pack has to name an oracle other hosts can reach, so
    some fixtures serve a routable address rather than loopback.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)])
    subject_alternative_name: x509.GeneralName
    try:
        subject_alternative_name = x509.IPAddress(ipaddress.ip_address(host))
    except ValueError:
        subject_alternative_name = x509.DNSName(host)
    now = dt.datetime.now(dt.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([subject_alternative_name]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path = (certificate_root or root) / "oracle-cert.pem"
    key_path = root / "oracle-key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path


@dataclass
class LibraryOracle:
    """Three reviewed tools: one read, one confirmed mutation, one that never returns."""

    content_digest: str
    attestation: dict[str, Any]
    slow_call_s: float = 30.0
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    opened: int = 0

    @property
    def tool_names(self) -> tuple[str, ...]:
        return ("borrow_book", "get_book_status", "rebuild_catalog_index")

    def identity(self) -> dict[str, str]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "oracle_id": ORACLE_ID,
            "oracle_version": ORACLE_VERSION,
            "content_digest": self.content_digest,
        }

    def open_session(self, fixtures: Mapping[str, Any] | None) -> str:
        self.opened += 1
        session_id = f"session-{self.opened}"
        books = {}
        for entry in (fixtures or {}).get("books", []):
            books[str(entry["book_id"])] = str(entry["status"])
        self.sessions[session_id] = {"books": books}
        return session_id

    def call(self, session_id: str, name: str, arguments: Mapping[str, Any]) -> Any:
        state = self.sessions[session_id]
        if name == "rebuild_catalog_index":
            time.sleep(self.slow_call_s)
            return {"status": "succeeded"}
        if name == "get_book_status":
            book_id = str(arguments.get("book_id", ""))
            if book_id not in state["books"]:
                return {"error": {"code": "not_found", "message": "unknown book"}}
            return {"status": state["books"][book_id]}
        if name == "borrow_book":
            book_id = str(arguments.get("book_id", ""))
            if book_id not in state["books"]:
                return {"error": {"code": "not_found", "message": "unknown book"}}
            if arguments.get("confirm") is not True:
                return {"status": "awaiting_confirmation", "book_id": book_id}
            state["books"][book_id] = "on_loan"
            return {"status": "succeeded", "book_id": book_id}
        return {"error": {"code": "unknown_tool", "message": name}}


@dataclass(frozen=True)
class RunningOracle:
    oracle: LibraryOracle
    base_url: str
    certificate_path: Path


def _handler(oracle: LibraryOracle) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return

        def _send(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _session_id(self) -> str:
            from urllib.parse import unquote

            return unquote(self.path.split("/")[3])

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path == "/v1/metadata":
                self._send(oracle.identity())
            elif self.path == "/v1/tools":
                self._send({"tools": list(oracle.tool_names)})
            elif self.path == "/v1/conformance":
                self._send(oracle.attestation)
            elif self.path.endswith("/state"):
                session = oracle.sessions.get(self._session_id())
                if session is None:
                    self._send({"error": "unknown session"}, 404)
                else:
                    self._send(session)
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/v1/sessions":
                session_id = oracle.open_session(payload.get("fixtures"))
                self._send({"session_id": session_id, "oracle": oracle.identity()})
            elif self.path.endswith("/calls"):
                session_id = self._session_id()
                if session_id not in oracle.sessions:
                    self._send({"error": "unknown session"}, 404)
                    return
                self._send(
                    oracle.call(
                        session_id,
                        payload["name"],
                        payload.get("arguments") or {},
                    )
                )
            else:
                self._send({"error": "not found"}, 404)

        def do_DELETE(self) -> None:  # noqa: N802
            oracle.sessions.pop(self._session_id(), None)
            self._send({})

    return Handler


@contextlib.contextmanager
def serve_library_oracle(
    oracle: LibraryOracle,
    *,
    root: Path,
    certificate_root: Path | None = None,
) -> Iterator[RunningOracle]:
    """Serve one oracle over TLS on a loopback port for the duration of the block."""
    certificate_path, key_path = write_localhost_certificate(
        root,
        certificate_root=certificate_root,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(oracle))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(certificate_path), keyfile=str(key_path))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield RunningOracle(
            oracle=oracle,
            base_url=f"https://127.0.0.1:{server.server_address[1]}",
            certificate_path=certificate_path,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
