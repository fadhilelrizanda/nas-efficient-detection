#!/usr/bin/env python3
"""Idempotent registration of the W&B MCP into the active Hermes profile.

Reads `WANDB_API_KEY` from the project `.env`, ensures `wandb` is in the
`mcp_servers:` block of `~/.hermes/profiles/<active>/config.yaml`, and
enables all 22 tools.

Safe to re-run: if the entry already exists, the script no-ops and
prints the current status. If `--force` is passed, it removes the
existing entry and re-adds it (useful when args or env change).

Usage:
    python3 scripts/register.py            # add if missing, no-op if present
    python3 scripts/register.py --force    # remove and re-add
    python3 scripts/register.py --status   # just show the current state
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Repo-relative defaults; override with env vars if needed.
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
    """Read WANDB_API_KEY from a dotenv file, tolerating unterminated quotes."""
    if not env_file.exists():
        raise SystemExit(f"env file not found: {env_file}")
    for line in env_file.read_text().splitlines():
        if line.startswith("WANDB_API_KEY"):
            _, _, raw = line.partition("=")
            raw = raw.strip()
            # strip matched wrapping quotes
            while len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
                raw = raw[1:-1]
            # strip any unmatched single quote
            if raw.startswith(('"', "'")):
                raw = raw[1:]
            if raw.endswith(('"', "'")):
                raw = raw[:-1]
            if raw:
                return raw
    raise SystemExit(f"WANDB_API_KEY not found in {env_file}")


def parse_wandb_block(config_path: Path) -> dict | None:
    """Return the parsed wandb entry from hermes config, or None if missing."""
    if not config_path.exists():
        return None
    text = config_path.read_text()
    m = re.search(r'^  ' + SERVER_LABEL + r':((?:\n(?!  \w).*)+)', text, re.MULTILINE)
    if not m:
        return None
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
    result["enabled"] = "enabled: true" in block
    return result


def hermes_mcp_list() -> list[dict]:
    """Parse `hermes mcp list` text output into a list of dicts."""
    out = subprocess.run(
        ["hermes", "mcp", "list"], check=True, text=True, capture_output=True,
    ).stdout
    rows = []
    for line in out.splitlines():
        # Format: "  wandb            uvx --from git+...      all  ✓ enabled"
        m = re.match(r'^\s+(\S+)\s+(.+?)\s+(all|none|\d+)\s+([✓✗])\s+(enabled|disabled)\s*$', line)
        if m:
            rows.append({
                "name": m.group(1),
                "transport": m.group(2).strip(),
                "tools": m.group(3),
                "status": m.group(5),
            })
    return rows


def add_server(env_value: str) -> None:
    """Run `hermes mcp add wandb ...` with the canonical args.

    Pipes `y` to accept the tool-enable prompt automatically.
    """
    # Build env value by concatenation so the secret doesn't sit next to a
    # redaction placeholder when this script is logged.
    env_arg = "WANDB_API_KEY=" + env_value
    cmd = [
        "hermes", "mcp", "add", SERVER_LABEL,
        "--command", "uvx",
        "--env", env_arg,
        "--args", "--from",
        "git+https://github.com/wandb/wandb-mcp-server",
        "wandb_mcp_server",
    ]
    print(f"[run] hermes mcp add {SERVER_LABEL} ...")
    proc = subprocess.run(cmd, input="y\ny\ny\n", text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"hermes mcp add failed with exit {proc.returncode}")


def remove_server() -> None:
    print(f"[run] hermes mcp remove {SERVER_LABEL}")
    proc = subprocess.run(
        ["hermes", "mcp", "remove", SERVER_LABEL],
        input="y\n", text=True, capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)


def cmd_status(args) -> int:
    entry = parse_wandb_block(args.config)
    if entry is None:
        print(f"status: '{SERVER_LABEL}' is NOT in {args.config}")
        return 1
    has_key = "WANDB_API_KEY" in entry["env"]
    masked = (
        f"{entry['env']['WANDB_API_KEY'][:8]}...{entry['env']['WANDB_API_KEY'][-4:]}"
        if has_key and len(entry["env"]["WANDB_API_KEY"]) >= 12
        else "(too short)"
    )
    print(f"status: '{SERVER_LABEL}' IS in {args.config}")
    print(f"  command: {entry.get('command')}")
    print(f"  args:    {entry.get('args')}")
    print(f"  enabled: {entry.get('enabled')}")
    print(f"  WANDB_API_KEY: {'set' if has_key else 'MISSING'} ({masked})")

    # cross-check with hermes's runtime view
    rows = hermes_mcp_list()
    rt = next((r for r in rows if r["name"] == SERVER_LABEL), None)
    if rt is None:
        print("  runtime: not loaded (start a new session to pick up config changes)")
    else:
        print(f"  runtime: {rt['status']}, tools={rt['tools']}")
    return 0


def cmd_register(args) -> int:
    key = read_wandb_key(args.env_file)
    print(f"key: len={len(key)}, prefix={key[:8]}..., suffix=...{key[-4:]}")

    existing = parse_wandb_block(args.config)
    if existing and not args.force:
        print(f"'{SERVER_LABEL}' already registered in {args.config}")
        print("re-run with --force to remove and re-add")
        return cmd_status(args)

    if existing and args.force:
        print(f"force: removing existing '{SERVER_LABEL}' entry first")
        remove_server()

    add_server(key)

    # verify
    new = parse_wandb_block(args.config)
    if new is None:
        raise SystemExit("registration reported success but entry is not in config")
    print(f"verified: '{SERVER_LABEL}' is now in {args.config}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--env-file", type=Path, default=None,
                   help="Path to the .env file. If omitted, searches a few likely locations "
                        "(REPO_ROOT, REPO_ROOT.parent, cwd, research-projects mirror).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"Hermes config.yaml (default: {DEFAULT_CONFIG})")
    p.add_argument("--force", action="store_true",
                   help="Remove and re-add the entry even if it exists")
    p.add_argument("--status", action="store_true",
                   help="Just print the current registration state and exit")
    args = p.parse_args()

    args.env_file = find_env_file(args.env_file)

    if args.status:
        return cmd_status(args)
    return cmd_register(args)


if __name__ == "__main__":
    sys.exit(main())
