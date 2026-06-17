# wandb-mcp

Persistent W&B (Weights & Biases) integration for Hermes in this repo.

Unlike `kaggle-mcp/` (which wraps a CLI we invoke ourselves), this directory
documents and operationalizes the **official** [`wandb-mcp-server`](https://github.com/wandb/wandb-mcp-server)
package so any operator or agent can re-register, validate, and use the
W&B tool surface from Hermes with one command.

## What is here

- `AGENT.md` — agent-facing guide: env contract, registration, validation,
  the gotchas we hit during initial install, and the exact `hermes mcp add`
  command. Read this before touching the registration.
- `README.md` — operator quick start (this file).
- `scripts/register.py` — idempotent wrapper around `hermes mcp add wandb`.
  Sources the key from the project `.env` and registers the server with all
  22 tools enabled. Safe to re-run.
- `scripts/probe.py` — end-to-end MCP auth probe. Spawns the same stdio
  command Hermes uses and runs a full `initialize` → `tools/list` →
  `tools/call list_entities_tool` round-trip. Use this to validate any
  W&B MCP install on any machine, not just the one that registered it.

## Quick start

```bash
# 1. Make sure WANDB_API_KEY is in the project .env
grep WANDB_API_KEY /home/fadhil/program/research-projects/nas-efficient-detection/.env

# 2. Register (idempotent — skips if 'wandb' is already in hermes config)
python3 wandb-mcp/scripts/register.py

# 3. Validate
python3 wandb-mcp/scripts/probe.py
```

If `probe.py` ends with `RESULT: OK` and prints an `entities` list, the
install is good. If it prints `RESULT: FAILED`, jump to the troubleshooting
section in `AGENT.md`.

## What this is NOT

- Not a re-implementation of the W&B MCP server. The official
  `wandb-mcp-server` package (pinned in `AGENT.md`) is the source of truth.
- Not a place for the project experiment code. That lives in `code/`.
- Not a place for downloaded artifacts. Those go in `kaggle-nb/output/`.

## Maintenance

The official `wandb-mcp-server` moves on its own release cadence. If a
new version changes the tool list, the framing, or the env contract,
update `AGENT.md` first, then `scripts/probe.py`, then this README.
