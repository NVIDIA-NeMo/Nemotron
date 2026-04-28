# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for nemo_runspec.execution helpers."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field

from nemo_runspec.execution import (
    _parse_netrc,
    materialize_podman_auth_from_enroot,
)


@dataclass
class _Result:
    ok: bool = True
    stdout: str = ""
    stderr: str = ""


@dataclass
class _ScriptedTunnel:
    """Fake tunnel that returns scripted file contents and records writes.

    ``files`` maps an enroot-credentials path to its contents (or ``None``
    if it should be reported missing). The tunnel intercepts the
    ``test -f X && cat X || true`` read pattern emitted by
    ``materialize_podman_auth_from_enroot`` and returns the corresponding
    content, and intercepts the ``printf %s <b64> | base64 -d > X`` write
    pattern to capture decoded contents in ``writes``.
    """

    files: dict[str, str | None]
    commands: list[str] = field(default_factory=list)
    writes: dict[str, str] = field(default_factory=dict)

    def run(self, cmd: str, hide: bool = False, warn: bool = False) -> _Result:
        self.commands.append(cmd)
        cat_match = re.search(r'test -f "([^"]+)" && cat "[^"]+" \|\| true', cmd)
        if cat_match:
            path = cat_match.group(1)
            content = self.files.get(path)
            if content is None:
                # Match real shell behaviour: with `|| true` the command
                # succeeds with empty stdout.
                return _Result(ok=True, stdout="")
            return _Result(ok=True, stdout=content)

        # ``shlex.quote`` only adds single quotes when needed (e.g. paths
        # with shell metachars), so accept both quoted and unquoted forms.
        write_match = re.search(
            r"printf %s (\S+) \| base64 -d > (\S+)",
            cmd,
        )
        if write_match:
            encoded = _unwrap_quotes(write_match.group(1))
            path = _unwrap_quotes(write_match.group(2))
            self.writes[path] = base64.b64decode(encoded).decode()

        return _Result(ok=True)


def _unwrap_quotes(s: str) -> str:
    """Strip a single matching pair of single or double quotes."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


# ---------------------------------------------------------------------------
# _parse_netrc
# ---------------------------------------------------------------------------


class TestParseNetrc:
    def test_single_machine(self):
        creds = _parse_netrc("machine nvcr.io login $oauthtoken password nvapi-secret")
        assert creds == {"nvcr.io": ("$oauthtoken", "nvapi-secret")}

    def test_multiple_machines_on_separate_lines(self):
        content = (
            "machine gitlab.example.com login alice password gl-token\n"
            "machine nvcr.io login $oauthtoken password nvapi-secret\n"
        )
        assert _parse_netrc(content) == {
            "gitlab.example.com": ("alice", "gl-token"),
            "nvcr.io": ("$oauthtoken", "nvapi-secret"),
        }

    def test_default_block_is_skipped(self):
        # ``default`` is netrc's catch-all entry; we don't surface it
        # because podman auth.json is keyed per-registry.
        content = (
            "machine nvcr.io login $oauthtoken password nvapi-secret\n"
            "default login fallback password fbk-token\n"
        )
        assert _parse_netrc(content) == {
            "nvcr.io": ("$oauthtoken", "nvapi-secret"),
        }

    def test_empty_string_returns_empty_dict(self):
        assert _parse_netrc("") == {}

    def test_ignores_extra_whitespace(self):
        content = "  machine\tnvcr.io  login   $oauthtoken\n  password   secret\n"
        assert _parse_netrc(content) == {"nvcr.io": ("$oauthtoken", "secret")}


# ---------------------------------------------------------------------------
# materialize_podman_auth_from_enroot
# ---------------------------------------------------------------------------


class TestMaterializePodmanAuth:
    NETRC = (
        "machine gitlab.example.com login alice password gl-token\n"
        "machine nvcr.io login $oauthtoken password nvapi-secret\n"
    )
    DEFAULT_PATH = "$HOME/.config/enroot/.credentials"

    def test_returns_none_when_credentials_missing(self):
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: None})
        assert materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth") is None
        assert tunnel.writes == {}

    def test_returns_none_when_no_target_registry_present(self):
        tunnel = _ScriptedTunnel(
            files={self.DEFAULT_PATH: "machine gitlab.example.com login a password b\n"},
        )
        assert (
            materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth") is None
        )
        assert tunnel.writes == {}

    def test_writes_auth_json_for_default_registry(self):
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: self.NETRC})

        path = materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth")

        assert path == "/lustre/cache/.auth/auth.json"
        assert path in tunnel.writes
        payload = json.loads(tunnel.writes[path])
        # Only nvcr.io should appear by default — other entries (gitlab)
        # must not leak into the build container.
        assert set(payload["auths"].keys()) == {"nvcr.io"}
        decoded = base64.b64decode(payload["auths"]["nvcr.io"]["auth"]).decode()
        assert decoded == "$oauthtoken:nvapi-secret"

    def test_respects_explicit_registry_allowlist(self):
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: self.NETRC})

        path = materialize_podman_auth_from_enroot(
            tunnel,
            "/lustre/cache/.auth",
            registries=("nvcr.io", "gitlab.example.com"),
        )

        assert path == "/lustre/cache/.auth/auth.json"
        payload = json.loads(tunnel.writes[path])
        assert set(payload["auths"].keys()) == {"nvcr.io", "gitlab.example.com"}

    def test_registry_match_is_case_insensitive(self):
        netrc = "machine NVCR.IO login $oauthtoken password nvapi-secret\n"
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: netrc})

        path = materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth")

        # Case-folding is applied to the match so the configured allowlist
        # tolerates the netrc entry's casing.
        payload = json.loads(tunnel.writes[path])
        assert set(payload["auths"].keys()) == {"NVCR.IO"}

    def test_writes_with_mode_0600(self):
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: self.NETRC})
        materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth")
        # The chmod is part of the same shell command as the write; assert
        # at least one issued command sets 0600 on the auth file.
        assert any(
            "chmod 600" in c and "/lustre/cache/.auth/auth.json" in c
            for c in tunnel.commands
        ), tunnel.commands

    def test_creates_out_dir(self):
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: self.NETRC})
        materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth")
        assert any(
            "mkdir -p" in c and "/lustre/cache/.auth" in c for c in tunnel.commands
        ), tunnel.commands

    def test_custom_credentials_path(self):
        custom = "/etc/enroot/credentials"
        tunnel = _ScriptedTunnel(files={custom: self.NETRC})
        path = materialize_podman_auth_from_enroot(
            tunnel,
            "/lustre/cache/.auth",
            enroot_credentials_path=custom,
        )
        assert path == "/lustre/cache/.auth/auth.json"
