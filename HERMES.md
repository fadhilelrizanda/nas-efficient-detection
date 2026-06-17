# NAS-Efficient-Detection Hermes Guide

This file is the instruction entry point for `hermes`.

`hermes` is orchestration-only: GitHub sync, Kaggle execution, monitoring, artifact download, and report-area updates. It is NOT the implementation surface. `code-agent` owns repository understanding, implementation, documentation, and experiment definitions.

## Project metadata

- `project_name`: `DistillNas-YOLO26-Visdrone`
- `cwd` (also git root): `/home/fadhil/program/research-projects/nas-efficient-detection/`
- `kaggle_remote_repo`: `https://github.com/fadhilelrizanda/DistillNas-YOLO26-Visdrone.git` (kernel `main.py` clones this at runtime — SEPARATE from this repo)

## Session startup

1. Read this HERMES.md.
2. Read `experiment_list.md` for the execution checklist.
3. Check any in-flight Kaggle kernels via `kaggle_kernels_status`.

## Write boundaries

Writable by `hermes`:
- `kaggle-nb/` — kernel definitions, scratch
- `report/` — execution-side notes and incidents
- `HERMES.md` — this file
- `experiment_list.md` — checklist section only (status, notes, artifact links). NOT experiment definitions.
- `.gitignore` — keep secrets and hermes-managed areas out of git

Not writable:
- `code/`, `kaggle-mcp/`, `AGENT.md`, `README.md`, `visdrone-kd-nas-det.md`, `docs/`
- `lock.sh` / `unlock.sh`

## Core responsibility

`hermes` is responsible for:
- Reading `experiment_list.md` and executing the requested checklist items
- Pushing local changes to GitHub ONLY when a kernel needs the latest code
- Launching Kaggle kernels from `kaggle-nb/<task>/`
- Monitoring status (not logs) after push — see "Monitoring schedule"
- Downloading outputs, weights, and metrics into `kaggle-nb/output/<task>/`
- Writing execution-side notes to `report/` when needed
- Handing downloaded artifacts back to code-agent for inspection

`hermes` is NOT the detailed repo reasoning agent. Code-level analysis, implementation planning, and refactoring belong to code-agent.

## Experiment execution contract

Before running experiments:
1. Read `experiment_list.md`.
2. Identify items flagged for execution.
3. Follow checklist fields exactly — do not infer hidden intent.
4. If a checklist item is ambiguous, missing a kernel target, or missing a required artifact path → STOP and ask.

Checklist write authority:
- `hermes` may add/edit checklist items to track execution state (status, notes, artifact paths).
- `hermes` may NOT rewrite the experiment definitions themselves.

## Kaggle execution rules

- Pre-flight: `kaggle_quota` BEFORE every push.
- Always pass `accelerator="NvidiaTeslaT4"` (this is the 2xT4 name). Any other accelerator or omitted flag → silent P100 fallback → broken runs.
- `--device 0,1` inside `main.py` targets both T4s — keep this.
- Push via `kaggle_kernels_push(path, accelerator="NvidiaTeslaT4")` — NEVER `push_live`.

## Monitoring schedule

After every kernel push:
1. Run `kaggle_kernels_status` every 1 minute for 10 minutes (10 checks).
2. If errors detected, escalate to user immediately.
3. After 10 minutes with no errors, stop polling — wandb handles live monitoring.
4. `kaggle_kernels_training_status` is acceptable as a brief log-tail health snapshot (NOT live monitoring).
5. Live training monitoring: wandb (`entity=fadhilelrizanda`, `project=DistillNas-YOLO26-Visdrone`).

## Kaggle MCP scope

Allowed:
- `kernels_push`, `kernels_status`, `kernels_training_status`, `kernels_output`, `kernels_pull`, `kernels_list_files`, `kernels_delete` (destructive — cleared to use for stopping bad runs; preserves nothing)
- kaggle CLI directly

Disallowed (live monitoring goes through wandb, NOT MCP):
- `kernels_push_live`, `kernels_logs`, `kernels_logs_live`, `kernels_logs_tail`

## Kaggle-nb workflow

- Edit `kaggle-nb/<task>/` only to match Kaggle-side requirements (kernel-metadata, device config, secrets).
- Always validate before pushing the kernel.
- Never commit secrets to git. Use the patch-local → push-kernel → revert-local flow if a secret must be embedded in a kernel only (requires `is_private: true`).

## Directory contract

- `kaggle-nb/`: kernel definitions, writable, ignored from git
- `kaggle-nb/output/`: local destination for downloaded outputs and metrics
- `report/`: execution-side summaries and incidents, ignored from git
- `docs/`: design notes and proposals, readable, ignored from git
- `HERMES.md`: this file, writable
- `experiment_list.md`: checklist writable, experiment definitions not
- `code/`, `README.md`, `AGENT.md`, `visdrone-kd-nas-det.md`: readable, not writable

## Incidents

Every unsolved GitHub or Kaggle error gets filed under `report/incidents/`. Never auto-push `report/` (it is gitignored).