# NAS-Efficient-Detection Code Agent Guide

This file is the instruction entry point for `code-agent`.

`code-agent` is the primary repository worker. It is responsible for detailed repo understanding, implementation work, documentation maintenance, experiment planning, and interpreting downloaded experiment results.

## Project metadata

- `project_name`: `DistillNas-YOLO26-Visdrone`

## Access model

`code-agent` has:

- read access to all repository directories
- write access to the main repository workspace except `report/`
- no write access to `report/`

This means `code-agent` may inspect files under `report/`, but must not create, edit, or reorganize report files there.

## Core responsibility

`code-agent` should:

- read the repository in detail before editing code or docs
- own implementation work in `code/`
- keep `kaggle-nb/` task folders aligned with repository code, with code-side task wrappers that match Kaggle task defaults
- maintain repository documentation such as `README.md`, `AGENT.md`, and files under `docs/`
- inspect downloaded outputs and metrics in `kaggle-nb/output/` and use them to drive further code changes
- write and maintain [experiment_list.md](/home/fadhil/program/research-projects/nas-efficient-detection/experiment_list.md) as the handoff document that tells `hermes` which experiments need to run
- write and maintain [comparisons/overall_experiment_comparison.md](/home/fadhil/program/research-projects/nas-efficient-detection/comparisons/overall_experiment_comparison.md) as the single-file experiment comparison tracker across benchmark, fine-tune, supernet, and student runs

`code-agent` should not be the agent used primarily for GitHub push/pull, Kaggle launch, remote run monitoring, or writing under `report/`. Those belong to `hermes`.

## Repository layout

### Root
- [visdrone-kd-nas-det.md](/home/fadhil/program/research-projects/nas-efficient-detection/visdrone-kd-nas-det.md): main research working document for the current direction
- [AGENT.md](/home/fadhil/program/research-projects/nas-efficient-detection/AGENT.md): this file, for `code-agent`
- [HERMES.md](/home/fadhil/program/research-projects/nas-efficient-detection/HERMES.md): orchestration guide for `hermes`
- [experiment_list.md](/home/fadhil/program/research-projects/nas-efficient-detection/experiment_list.md): experiment handoff checklist written by `code-agent` and consumed by `hermes`
- [comparisons/overall_experiment_comparison.md](/home/fadhil/program/research-projects/nas-efficient-detection/comparisons/overall_experiment_comparison.md): single-file summary of experiment results and placeholders across benchmark, fine-tune, supernet, and student stages
- `docs/`: project documentation and working notes, organized into `learn/`, `proposed/`, `base-knowledge/`, and `other/`
- `code/`: canonical experiment and model code
- `kaggle-nb/`: Kaggle execution wrappers and task directories
- `kaggle-nb/helpers/`: local Kaggle workflow helpers
- `kaggle-nb/output/`: local mirror for downloaded Kaggle outputs and metrics
- `comparisons/`: experiment comparison notes maintained by `code-agent`
- `report/`: report-facing area that `code-agent` may read but must not modify

### Utility subproject
- [kaggle-mcp/server.py](/home/fadhil/program/research-projects/nas-efficient-detection/kaggle-mcp/server.py): MCP server wrapping the Kaggle CLI
- [kaggle-mcp/AGENT.md](/home/fadhil/program/research-projects/nas-efficient-detection/kaggle-mcp/AGENT.md): Kaggle MCP subproject instructions

## Project focus

The active research direction is:

- distillation-aware neural architecture search for lightweight YOLO26 students
- teacher: YOLO26x fine-tuned on VisDrone-DET
- search target: VisDrone-DET small-object detection under edge latency constraints
- core comparison: KD-aware NAS versus KD-blind NAS at matched latency
- dataset scope: `VisDrone-DET` only for active detection research and implementation unless the user explicitly says otherwise

Dataset policy:

- default to `VisDrone-DET` only
- do not switch to `VisDrone-MOT` or `VisDrone-VID` for training, evaluation, or documentation unless explicitly requested
- if older files or utilities reference MOT, treat them as legacy or auxiliary material rather than the active dataset direction

## Kaggle-first execution contract

This repository executes experiments on Kaggle. Because of that:

- code-side task entrypoints in `code/` must mirror Kaggle task folders when the runtime defaults differ by task
- if a Kaggle task uses a specific batch size, GPU layout, run name, report name, or W&B tagging policy, the corresponding `code/` wrapper must use the same defaults
- do not let local code defaults drift away from the Kaggle task that actually runs them
- shared logic belongs in Python modules such as `code/visdrone_det/`; task-specific execution defaults belong in the task wrapper that matches the Kaggle folder

Current pairings:

- `kaggle-nb/visdrone-det-yolo26x-benchmark/` <-> `code/make_visdrone_det_yolo26x_benchmark.py`
- `kaggle-nb/visdrone-det-yolo26x-finetuning/` <-> `code/make_visdrone_det_yolo26x_finetuning.py`

For the current fine-tuning task, the canonical execution defaults are:

- batch size `8`
- training device `0,1` for `2xT4`
- W&B-enabled training unless explicitly disabled

## Experiment handoff contract

`code-agent` must use `experiment_list.md` to communicate runnable work to `hermes`.

`code-agent` must use `comparisons/overall_experiment_comparison.md` to keep one rolling summary of experiment outcomes across the major stages of the project.

Rules for `experiment_list.md`:

- `code-agent` owns the content and keeps it up to date
- every experiment entry should be actionable without extra interpretation
- every experiment entry should have a checklist status so `hermes` can track execution progress
- if a run depends on a specific branch, commit, kernel folder, dataset assumption, or output path, write that explicitly
- if a run should not be executed yet, mark it clearly as blocked or pending
- do not use `report/` as the substitute for this handoff because `code-agent` cannot write there

Rules for `comparisons/overall_experiment_comparison.md`:

- keep all major experiment families in one file rather than scattering comparison notes across many markdown files
- treat the current YOLO26x benchmark as the initial baseline unless a stronger baseline replaces it
- add future teacher fine-tune, supernet, and student results into the same tracker
- keep summary-table metrics synchronized with the detailed experiment sections below them
- source results from checked-in local outputs such as files under `kaggle-nb/output/` whenever available

Minimum content per experiment item:

- experiment name
- purpose or hypothesis
- exact Kaggle task or local orchestration target
- required code state or GitHub ref if relevant
- required environment or accelerator
- expected outputs or metrics to download
- checklist state such as `todo`, `running`, `done`, or `blocked`

## Code-agent workflow

When working in this repository:

1. Read [visdrone-kd-nas-det.md](/home/fadhil/program/research-projects/nas-efficient-detection/visdrone-kd-nas-det.md) before making research-facing changes.
2. Determine whether the task belongs in `code/`, `docs/`, `kaggle-nb/`, or `kaggle-mcp/`.
3. Default to local repository reasoning first. Inspect the real code and docs before changing workflow assumptions.
4. Put reusable training, search, evaluation, and dataset logic in `code/`, not in Kaggle kernel folders.
5. Keep `kaggle-nb/` focused on entry scripts, metadata, setup glue, and task packaging.
6. When a task has execution-specific defaults, express them in a code-side wrapper that mirrors the Kaggle task rather than hiding them in ad hoc notes.
7. If Kaggle outputs or metrics already exist locally, inspect them before changing implementation assumptions.
8. Update `experiment_list.md` whenever new experiments need to be run, re-run, cancelled, or compared.
9. Update [comparisons/overall_experiment_comparison.md](/home/fadhil/program/research-projects/nas-efficient-detection/comparisons/overall_experiment_comparison.md) when new benchmark, fine-tune, supernet, or student results are available locally.
10. If the task requires GitHub sync, Kaggle execution, log monitoring, artifact download, or report writing, hand that orchestration off to `hermes` under [HERMES.md](/home/fadhil/program/research-projects/nas-efficient-detection/HERMES.md).

## Editing guidance

- Keep `code/` as the canonical source of logic.
- Keep `docs/` for human-readable project notes, references, and research support material.
- Keep `comparisons/overall_experiment_comparison.md` as the canonical one-file experiment comparison ledger.
- Prefer placing docs by purpose: `learn/` for study notes, `proposed/` for ideas and plans, `base-knowledge/` for theory references, and `other/` for supporting artifacts.
- Keep new experiment defaults aligned to `VisDrone-DET` unless directed otherwise.
- Keep Kaggle wrappers thin and avoid duplicating large implementation blocks into `kaggle-nb/`.
- When task defaults differ, make the code wrapper match the Kaggle task exactly for batch size, GPU selection, reports, and other execution-critical defaults.
- Do not write into `report/`; treat it as read-only from the `code-agent` side.
- When repository structure changes, update this file, `HERMES.md`, and the top-level `README.md` if the access model or handoff flow changed.
- Do not rewrite `kaggle-mcp/` to match the experiment code layout; it is a utility subproject.

## Known state

- `code/` is the intended main codebase.
- `kaggle-nb/` is the task-scoped runner layer.
- `code/` now includes task-specific wrappers that mirror Kaggle execution folders when defaults differ.
- `kaggle-nb/helpers/` supports local Kaggle workflows.
- `kaggle-nb/output/` stores downloaded remote outputs and metrics.
- `comparisons/overall_experiment_comparison.md` is the canonical single-file tracker for cross-experiment result comparison.
- `report/` exists and may be read by `code-agent`, but not modified by it.
- `code-agent` is the detailed repo worker; `hermes` is the orchestration and report-writing worker.
