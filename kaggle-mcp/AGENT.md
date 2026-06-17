# Kaggle MCP Agent Guide

This directory contains a single MCP server: [server.py](/home/fadhil/program/research-projects/nas-efficient-detection/kaggle-mcp/server.py).
Its job is simple: expose Kaggle CLI commands as MCP tools with profile-based token injection.

Within the larger repository workflow, this directory exists mainly to support `hermes` orchestration. It is not the main location for repository implementation work.

## What matters

- Source of truth: `server.py`
- Test payload for kernel push/log polling: [dummy-kernel/kernel-metadata.json](/home/fadhil/program/research-projects/nas-efficient-detection/kaggle-mcp/dummy-kernel/kernel-metadata.json) and [dummy-kernel/main.py](/home/fadhil/program/research-projects/nas-efficient-detection/kaggle-mcp/dummy-kernel/main.py)
- Runtime model: every tool shells out to `kaggle ...` and returns plain text

## What this server does

- Wraps Kaggle competitions, datasets, kernels, models, config, auth, and profile actions.
- Stores named Kaggle API token profiles in `~/.kaggle/profiles.json`.
- Injects the active profile token into subprocess calls through `KAGGLE_API_TOKEN`.
- Adds live training-monitoring tools that follow kernel logs with `kaggle kernels logs --follow` and summarize condition, epoch, and loss.

## Default agent rules

1. Treat `server.py` as the only implementation file that matters.
2. Use profile tools first if a request depends on a specific Kaggle account.
3. Treat this MCP server as an orchestration surface for `hermes`, not as a replacement for detailed repo analysis by `code-agent`.
4. For any training run, default to `kaggle_kernels_push_live(...)`, not `kaggle_kernels_push(...)`.
5. After a kernel is pushed, default to live monitoring immediately with `kaggle kernels logs --follow` via `kaggle_kernels_logs_live(...)`.
6. Use `kaggle_kernels_logs_live(...)` when the agent must inspect the run in real time; it should invoke `kaggle kernels logs --follow`.
7. Use `kaggle_kernels_training_status(...)` when the agent needs a fresh snapshot of health, epoch, and loss without a long follow window.
8. Use `kaggle_kernels_logs_tail(...)` only for cheap text-only polling.
9. Keep downloaded outputs inside an explicit path when a tool writes files.

## Required kernel workflow

When the user asks to run training on Kaggle, the agent should use this sequence by default:

1. `kaggle_kernels_push_live(path="...")`
2. Read the returned `condition`, `status`, `current_epoch`, and `current_loss`
3. If the returned condition is `healthy` or `starting`, continue monitoring with `kaggle_kernels_logs_live(kernel="owner/slug")`, which should invoke `kaggle kernels logs --follow`
4. If a cheaper refresh is enough, use `kaggle_kernels_training_status(kernel="owner/slug")`
5. If the condition becomes `unhealthy` or `stopped`, report that immediately

Do not push a training kernel and then stop without checking logs.

Do not move core implementation logic into this directory. `kaggle-mcp/` only exposes orchestration controls; `code/` remains the source of truth for experiment logic.

## Health interpretation

The live-monitoring tools try to classify the run into one of these states:

- `healthy`: training appears to be progressing or the kernel is running normally
- `starting`: the run has started but logs are still sparse or empty
- `completed`: the run appears finished successfully
- `stopped`: the run appears cancelled, killed, or stopped early
- `unhealthy`: the logs contain failure signals such as traceback, exception, or error
- `unknown`: there is not enough signal yet to classify the run

## Metric extraction

The monitoring tools try to extract these values from logs:

- `current_epoch`
- `current_loss`
- `epoch_line`
- `loss_line`

These are best-effort regex extractions from plain-text logs. If the training script uses unusual logging formats, values may remain `unknown`.

## High-value workflows

### 1. Verify auth state

Use these first when the user mentions account problems or multiple Kaggle users:

- `kaggle_profiles_list()`
- `kaggle_profile_show()`
- `kaggle_profile_use(name=...)`

If no profiles exist, the server tries to bootstrap a `default` profile from:

1. `KAGGLE_API_TOKEN`
2. `~/.kaggle/access_token`

### 2. Search before downloading

Typical sequence:

1. `kaggle_datasets_list(...)` or `kaggle_competitions_list(...)`
2. `kaggle_datasets_files(...)` or `kaggle_competitions_files(...)`
3. `kaggle_datasets_download(...)` or `kaggle_competitions_download(...)`

Do not download first and inspect later.

### 3. Push and monitor a training kernel

Use [dummy-kernel](/home/fadhil/program/research-projects/nas-efficient-detection/kaggle-mcp/dummy-kernel) when you need a known-good smoke test.

Recommended sequence:

1. `kaggle_kernels_push_live(path=".../dummy-kernel")`
2. Read the returned `condition`, `status`, `current_epoch`, and `current_loss`
3. If needed, call `kaggle_kernels_logs_live(kernel="owner/slug")` again for another `kaggle kernels logs --follow` window
4. If needed, call `kaggle_kernels_training_status(kernel="owner/slug")` for a cheaper snapshot

### 4. Snapshot monitoring rules

Use `kaggle_kernels_logs_tail(...)` when you only need recent text.
Use `kaggle_kernels_training_status(...)` when you need recent text plus parsed health, epoch, and loss.

## Tool groups

- Competitions: list, files, download, submit, submissions, leaderboard
- Datasets: list, files, download, create, version, status, metadata
- Kernels: list, list_files, push, push_live, pull, output, status, logs, logs_live, logs_tail, training_status, delete
- Models: list, get, instances_list, instances_versions_list, instances_versions_download
- Config/Auth: config_view, config_set, config_unset, quota, print_access_token, revoke
- Profiles: list, add, remove, use, show

## Constraints and caveats

- Tool output is plain text from the Kaggle CLI, not structured JSON.
- Errors may come back from either stdout or stderr; the server normalizes them into a string.
- Live follow calls are bounded by `follow_timeout`.
- When a live follow call times out, partial logs collected so far are still returned and summarized.
- Downloads and pulls write to the local filesystem when a path is supplied.
- Profile state is global to the server process because the active profile is stored in memory and persisted on disk.

## Launch

Run the server directly:

```bash
python server.py
```

The entrypoint is:

```python
def main() -> None:
    mcp.run()
```

## If you refine this further

- Keep agent instructions in this file only.
- Do not add another overlapping guide like `CLAUDE.md`.
- Do not commit secrets, `.env` files, or `__pycache__` artifacts.
