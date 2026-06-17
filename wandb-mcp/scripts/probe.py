#!/usr/bin/env python3
"""End-to-end MCP auth probe for the wandb server.

Spawns the same stdio command Hermes uses and runs a full MCP handshake:

  1. initialize
  2. notifications/initialized
  3. tools/list
  4. tools/call list_entities_tool

Exits 0 on full success, 1 on any failure. Use this to validate that a
W&B MCP install is *really* working — not just that `hermes mcp test`
reports a green checkmark (that only confirms the transport).

Usage:
    python3 scripts/probe.py
    python3 scripts/probe.py --env-file /path/to/.env
    python3 scripts/probe.py --config  /path/to/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes" / "profiles" / "nas-det"))
DEFAULT_CONFIG = HERMES_HOME / "config.yaml"
SERVER_LABEL = "wandb"


def find_env_file(explicit: Path | None) -> Path:
    """Locate WANDB_API_KEY's .env file.

    Search order:
      1. explicit path (from --env-file or WANDB_ENV_FILE)
      2. REPO_ROOT/.env
      3. REPO_ROOT.parent/.env  (covers the research-projects/<name>/.env
         layout when the script lives in the git-checkout mirror)
      4. cwd/.env
      5. ~/program/research-projects/<repo-name>/.env
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(REPO_ROOT / ".env")
    candidates.append(REPO_ROOT.parent / ".env")
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path.home() / "program" / "research-projects" / REPO_ROOT.name / ".env")
    seen: set[Path] = set()
    for c in candidates:
        try:
            resolved = c.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if c.exists():
            return c
    tried = "\n".join(f"  - {c}" for c in candidates)
    raise SystemExit(
        "could not locate .env with WANDB_API_KEY. Tried:\n"
        f"{tried}\n"
        "Pass --env-file /path/to/.env to point at the right file."
    )


def read_wandb_key(env_file: Path) -> str:
    if not env_file.exists():
        raise SystemExit(f"env file not found: {env_file}")
    for line in env_file.read_text().splitlines():
        if line.startswith("WANDB_API_KEY"):
            _, _, raw = line.partition("=")
            raw = raw.strip()
            while len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
                raw = raw[1:-1]
            if raw.startswith(('"', "'")):
                raw = raw[1:]
            if raw.endswith(('"', "'")):
                raw = raw[:-1]
            if raw:
                return raw
    raise SystemExit(f"WANDB_API_KEY not found in {env_file}")


def parse_wandb_block(config_path: Path) -> dict:
    if not config_path.exists():
        raise SystemExit(f"hermes config not found: {config_path}")
    text = config_path.read_text()
    m = re.search(r'^  ' + SERVER_LABEL + r':((?:\n(?!  \w).*)+)', text, re.MULTILINE)
    if not m:
        raise SystemExit(
            f"'{SERVER_LABEL}' not in {config_path}. "
            f"Run scripts/register.py first."
        )
    block = m.group(1)
    result: dict = {"args": [], "env": {}}
    in_args = in_env = False
    cm = re.search(r'^\s+command:\s*(.+)$', block, re.MULTILINE)
    if cm:
        result["command"] = cm.group(1).strip()
    for line in block.splitlines():
        if re.match(r'^\s+args:\s*$', line):
            in_args, in_env = True, False
            continue
        if re.match(r'^\s+env:\s*$', line):
            in_env, in_args = True, False
            continue
        if in_args:
            am = re.match(r'^\s+-\s*(.+?)\s*$', line)
            if am:
                result["args"].append(am.group(1).strip().strip("'\""))
            else:
                in_args = False
        if in_env:
            kv = re.match(r'^\s+([A-Z_][A-Z0-9_]*):\s*(.*?)\s*$', line)
            if kv:
                val = kv.group(2)
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                result["env"][kv.group(1)] = val
            else:
                in_env = False
    if "command" not in result or "wandb_mcp_server" not in " ".join(result["args"]):
        raise SystemExit(
            f"'{SERVER_LABEL}' entry in {config_path} looks malformed. "
            f"Got: command={result.get('command')!r}, args={result.get('args')!r}"
        )
    return result


class MCPClient:
    """Minimal stdio JSON-RPC client for the W&B MCP server.

    The W&B server uses line-delimited JSON on stdout and mixes
    server-pushed notifications with JSON-RPC responses. We must filter
    by `id`, not assume "first line wins".
    """

    def __init__(self, cmd: list[str], env: dict[str, str]):
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env={**os.environ, **env},
            text=False, bufsize=0,
        )
        self._buf = b""
        import threading
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self):
        for line in iter(self.proc.stderr.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            print(f"  [stderr] {text}", flush=True)

    def send(self, msg: dict) -> None:
        body = (json.dumps(msg) + "\n").encode("utf-8")
        self.proc.stdin.write(body)
        self.proc.stdin.flush()
        method = msg.get("method") or ("id=" + str(msg.get("id")))
        print(f"  [->] {method}", flush=True)

    def _read_line(self, timeout: float) -> str | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if b"\n" in self._buf:
                line, _, self._buf = self._buf.partition(b"\n")
                return line.decode("utf-8", errors="replace")
            r, _, _ = select.select([self.proc.stdout], [], [], 0.5)
            if not r:
                continue
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                return None
            self._buf += chunk
        return None

    def recv(self, expected_id, timeout: float = 30) -> dict | None:
        """Read lines until one has the expected `id`, or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line(max(1.0, deadline - time.time()))
            if line is None:
                return None
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [skip-parse] {line[:80]}", flush=True)
                continue
            if "method" in msg and "id" not in msg:
                # server-pushed notification, skip
                kind = msg.get("method", "?").split("/")[-1]
                print(f"  [skip-notif] {kind}", flush=True)
                continue
            if msg.get("id") == expected_id:
                return msg
            print(f"  [skip-id {msg.get('id')}] (wanted {expected_id})", flush=True)
        return None

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--env-file", type=Path, default=None,
                   help="Path to the .env file. If omitted, searches a few likely locations "
                        "(REPO_ROOT, REPO_ROOT.parent, cwd, research-projects mirror).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--timeout-init", type=float, default=20.0)
    p.add_argument("--timeout-tools", type=float, default=15.0)
    p.add_argument("--timeout-call", type=float, default=30.0)
    args = p.parse_args()
    args.env_file = find_env_file(args.env_file)

    key = read_wandb_key(args.env_file)
    print(f"key: len={len(key)}, prefix={key[:8]}..., suffix=...{key[-4:]}")
    entry = parse_wandb_block(args.config)
    print(f"cmd: {entry['command']} {' '.join(entry['args'])}")
    print(f"env: {list(entry['env'].keys())}")

    client = MCPClient(
        [entry["command"], *entry["args"]],
        {**entry["env"], "WANDB_API_KEY": key},
    )
    failed = False

    print("\nstep 1: initialize", flush=True)
    client.send({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "wandb-mcp-probe", "version": "1.0"},
        },
    })
    r = client.recv(1, timeout=args.timeout_init)
    if not r:
        print("INIT FAILED (timeout)"); failed = True
    else:
        srv = r.get("result", {}).get("serverInfo", {})
        print(f"  ok: server={srv.get('name')} v={srv.get('version')}", flush=True)

    if not failed:
        print("\nstep 2: notifications/initialized", flush=True)
        client.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        print("\nstep 3: tools/list", flush=True)
        client.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        r = client.recv(2, timeout=args.timeout_tools)
        if not r:
            print("TOOLS/LIST FAILED (timeout)"); failed = True
        else:
            tools = r.get("result", {}).get("tools", [])
            print(f"  ok: {len(tools)} tools", flush=True)
            for t in tools[:6]:
                print(f"    - {t['name']}")
            if len(tools) > 6:
                print(f"    ... and {len(tools) - 6} more")

    if not failed:
        print("\nstep 4: tools/call list_entities_tool", flush=True)
        client.send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "list_entities_tool", "arguments": {}},
        })
        r = client.recv(3, timeout=args.timeout_call)
        if not r:
            print("CALL FAILED (timeout)"); failed = True
        else:
            res = r.get("result", {})
            is_error = res.get("isError", False)
            print(f"  isError={is_error}", flush=True)
            for c in res.get("content", []):
                if c.get("type") == "text":
                    print(f"  text: {c.get('text', '')[:500]}", flush=True)
            if is_error:
                failed = True

    client.close()
    print(f"\nRESULT: {'FAILED' if failed else 'OK'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
