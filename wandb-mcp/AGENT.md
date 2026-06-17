# wandb-mcp — Agent Guide

This directory exists to make the W&B (Weights & Biases) MCP integration
in this repo **persistent and reproducible**. The actual MCP server is the
official [`wandb-mcp-server`](https://github.com/wandb/wandb-mcp-server)
package — we do not maintain a fork here. What we *do* maintain is:

1. The exact registration command (so any machine can re-create the
   Hermes entry from scratch).
2. The env-var contract (where the API key lives, who reads it, who must
   not commit it).
3. A reusable auth probe that confirms the install is actually working
   end-to-end — not just that the stdio handshake completes.
4. The gotchas we hit on first install, so the next person (or agent)
   does not re-discover them.

## Source of truth

- Official server: <https://github.com/wandb/wandb-mcp-server> (MIT,
  Python 3.11+, version pinned at install time via `uvx --from`).
- Docs: <https://docs.wandb.ai/platform/mcp-server>.
- This directory is to W&B MCP what `kaggle-mcp/` is to Kaggle MCP: a
  thin operational wrapper, not a competing implementation.

## Registration (what `hermes mcp add` actually saves)

The Hermes entry under `mcp_servers:` in `~/.hermes/profiles/<profile>/config.yaml`:

```yaml
wandb:
  command: uvx
  args:
    - --from
    - git+https://github.com/wandb/wandb-mcp-server
    - wandb_mcp_server
  env:
    WANDB_API_KEY: "<your key here>"
  enabled: true
```

`scripts/register.py` writes this exact shape. Re-running it is safe — it
checks for an existing entry and only adds when missing.

## Environment contract

| Source | Variable | Where it is read | Notes |
|--------|----------|------------------|-------|
| `.env` (project root) | `WANDB_API_KEY` | `scripts/register.py` and `scripts/probe.py` | New W&B keys are 86 chars, format `wandb_v1_...`. Older 40-char hex keys are often rejected with "relogin required" — rotate if you see that. |

The key is passed to the stdio child process via the `env:` block of the
Hermes MCP entry. We never pass it on the command line and we never commit
it. The project `.gitignore` already excludes `.env`.

## Available tools (22)

After registration, `hermes mcp list` should show `wandb` with `all` tools
enabled. The full tool surface:

- `query_weave_traces_tool` — query LLM traces
- `count_weave_traces_tool` — count traces with filters
- `resolve_trace_roots_tool` — batch-resolve root spans
- `query_wandb_tool` — GraphQL against the W&B Models API
- `create_wandb_report_tool` — build W&B Reports with charts
- `log_analysis_to_wandb` — log computed metrics to W&B
- `list_entities_tool` — list the W&B entities the key can see
- `query_wandb_entity_projects` — list projects under an entity
- `list_wandb_automations_tool` — list W&B Automations
- `list_wandb_integrations_tool` — list Slack/webhook integrations
- `infer_trace_schema_tool` — discover schema of Weave traces
- `search_wandb_docs_tool` — search W&B docs
- `get_run_history_tool` — sampled time-series of a run
- `list_registries_tool` — list model registries
- `list_registry_collections_tool` — list collections in a registry
- `list_artifact_versions_tool` — list versions of an artifact
- `get_artifact_details_tool` — full metadata for an artifact version
- `compare_artifact_versions_tool` — diff two artifact versions
- `compare_runs_tool` — diff two W&B runs
- `summarize_evaluation_tool` — aggregate Weave evaluation results
- `diagnose_run_tool` — health check on a W&B run
- `probe_project_tool` — discover a project's structure

## Validation

`hermes mcp test wandb` confirms the transport and tool discovery, but
**not** the API key. To validate auth end-to-end, run
`scripts/probe.py`. It performs the full MCP handshake and calls
`list_entities_tool`. A successful run prints something like:

```
step 1: initialize      → server=weave-mcp-server v=<x.y.z>     OK
step 2: notifications/initialized                              OK
step 3: tools/list       → 22 tools                            OK
step 4: list_entities_tool → isError=False                     OK
   {"entities": [{"name": "...", "type": "user"}], "count": N}
RESULT: OK
```

Any other outcome is a real failure — do not assume the install is good
just because `hermes mcp list` shows the green checkmark.

## Gotchas (the ones that cost us time)

1. **W&B MCP uses line-delimited JSON on stdout, not Content-Length
   framing.** The MCP spec mentions both, but the official W&B server
   only speaks LDJSON in stdio mode. If you write a custom client, send
   one JSON object per line and read one JSON object per line.
2. **The server mixes server-pushed notifications and JSON-RPC responses
   on the same stdout stream.** A notification like
   `{"severity": "INFO", "message": "Analytics ready: ..."}` arrives
   *before* the `initialize` response. Clients must filter by `id`,
   not by "first line wins".
3. **`hermes mcp add` stores multiple `--args` as literal strings** if
   you repeat the flag. The correct pattern is a single `--args` followed
   by all positional values, with `--command` and `--env` *before* it:
   ```bash
   hermes mcp add wandb \
     --command uvx \
     --env "WANDB_API_KEY=$KEY" \
     --args --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
   ```
4. **First `uvx --from git+...` launch is slow** (git clone + full pip
   build of the package and its deps). 30-90s on a cold cache. Subsequent
   launches are fast. `hermes mcp test` may time out on the very first
   try; rerun once the uvx cache is warm.
5. **The new W&B key format is 86 chars** starting with `wandb_v1_...`.
   Older 40-char hex keys are increasingly rejected with
   "Invalid W&B API key: relogin required". Generate a new one at
   <https://wandb.ai/authorize> if you see that message.
6. **Do not commit the `.env`.** The project root `.gitignore` already
   excludes it, but worth restating here: the key is a secret, the
   `wandb-mcp/scripts/*.py` files are not.
7. **Quoting in `.env`:** unterminated quotes (`WANDB_API_KEY="abc...`)
   are tolerated by `dotenv` but not by strict YAML parsers. The
   `scripts/probe.py` and `scripts/register.py` both fall back to
   stripping any unmatched leading/trailing quote.

## Troubleshooting flow

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `hermes mcp test wandb` → `Connection closed` | uvx cold cache OR wrong args | Rerun `register.py`, then `hermes mcp test wandb` again (cache is now warm) |
| Server stderr: `Invalid W&B API key: relogin required` | Key is revoked or in old hex format | Get a new key at wandb.ai/authorize and update `.env` |
| `probe.py` → `RESULT: FAILED` at step 1 | Transport issue | Check `hermes mcp list`; if `wandb` shows `disabled`, rerun `register.py` |
| `probe.py` → `RESULT: FAILED` at step 4 with `isError=True` | Auth fine but the entity is restricted | Verify the key is for the expected W&B account; check wandb.ai/authorize |
| `probe.py` hangs at any step | Server crashed mid-call | Check the `[stderr]` lines in the probe output for tracebacks |

## When to update this directory

- Tool list changed in a new `wandb-mcp-server` release → update the
  "Available tools" section above.
- Registration command changed → update the YAML block and
  `scripts/register.py` in lockstep.
- A new gotcha was discovered → append it under "Gotchas" with the
  symptom and the fix.
- The probe fails for a new reason → add a row to the troubleshooting
  table.

Do **not** add W&B experiment code here. That belongs in `code/`.
