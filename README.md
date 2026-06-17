# DistillNas-YOLO26-Visdrone

This repository is the main workspace for DistillNas YOLO26 experiments on VisDrone, with a strict split between local implementation and remote orchestration.

## Operating model

- `code-agent` is the primary repo worker. It reads the repository in detail, edits implementation, maintains docs, updates Kaggle payloads, and keeps the local structure coherent.
- `hermes` is orchestration-only. It should be used for pushing and pulling GitHub state, launching Kaggle training or evaluation jobs, monitoring logs, and downloading outputs or metrics.
- Remote execution must not become the main development surface. The canonical implementation stays in `code/`, while Kaggle wrappers stay thin and task-specific.
- Experiment logging now uses `wandb` (Weights & Biases) as the canonical run tracker, with live epoch metrics also mirrored to Kaggle stdout.

## Repository layout

- `code/`: canonical experiment and utility code
- `docs/`: research documentation organized into `learn/`, `proposed/`, `base-knowledge/`, and `other/`
- `kaggle-nb/`: Kaggle task directories and kernel payloads
- `kaggle-nb/helpers/`: local helper scripts for Kaggle orchestration
- `kaggle-nb/output/`: downloaded outputs and metrics from Kaggle runs
- `kaggle-mcp/`: MCP wrapper around the Kaggle CLI for orchestration workflows

## Current checked-in utilities

- `code/datasets/`: VisDrone MOT preview tooling used as auxiliary dataset utility material
- `code/visdrone_det/`: VisDrone-DET `YOLO26x` benchmark pipeline for train/val/test evaluation, latency profiling, GPU profiling, `wandb` logging, and test-challenge video export
- `kaggle-nb/visdrone-det-yolo26x-benchmark/`: Kaggle runner that pulls the GitHub repo and executes the main detection pipeline on GPU

## Workflow contract

1. Implement and refine reusable logic locally in `code/`.
2. Keep Kaggle task folders under `kaggle-nb/` as thin execution wrappers around repository code.
3. Use `hermes` only to orchestrate GitHub sync, Kaggle runs, log monitoring, and artifact download.
4. Pull metrics and outputs back into `kaggle-nb/output/<task>/` for local inspection.

See [AGENT.md](/home/fadhil/program/research-projects/nas-efficient-detection/AGENT.md) for `code-agent`, [HERMES.md](/home/fadhil/program/research-projects/nas-efficient-detection/HERMES.md) for `hermes`, [docs/README.md](/home/fadhil/program/research-projects/nas-efficient-detection/docs/README.md) for the research-docs structure, and `code/README.md` for the implementation layout.
