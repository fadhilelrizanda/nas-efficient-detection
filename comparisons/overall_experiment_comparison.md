# Overall Experiment Comparison

This file tracks the main experiments for this repository in one place. The current checked-in YOLO26x benchmark is the initial baseline, and future entries can cover teacher fine-tuning, supernet search, and distilled student models.

Current baseline source:
- `kaggle-nb/output/visdrone-det-yolo26x-benchmark/outputs/visdrone_det/reports/benchmark_report.json`
- Report created at: `2026-06-15T10:47:12.182435+00:00`

## Experiment Summary Table

| Experiment | Stage | Model / Artifact | Val mAP50 | Val mAP50-95 | Precision | Recall | Median latency (ms) | Mean FPS | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26x benchmark baseline | benchmark | `yolo26x.pt` | `0.2893` | `0.1703` | `0.4107` | `0.3249` | `44.2911` | `22.5183` | `done` | `1 epoch smoke benchmark` |
| YOLO26x fine-tune | fine-tune | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `planned` | `full teacher recipe placeholder` |
| Supernet search | supernet | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `planned` | `search and Pareto frontier placeholder` |
| Student model | student | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `planned` | `KD/NAS student placeholder` |

## Current Experiment Detail: YOLO26x Benchmark Baseline

### Run configuration

| Field | Value |
| --- | --- |
| Model | `yolo26x.pt` |
| Epochs | `1` |
| Image size | `640` |
| Batch size | `2` |
| Device | `cuda:0` |
| Latency split | `val` |
| Latency images | `32` |
| Warmup images | `10` |

### Benchmark summary

| Metric | Value |
| --- | --- |
| Val mAP50 | `0.2893` |
| Val mAP50-95 | `0.1703` |
| Val precision | `0.4107` |
| Val recall | `0.3249` |
| Median latency (ms/image) | `44.2911` |
| Mean latency (ms/image) | `44.4084` |
| Mean throughput (FPS) | `22.5183` |
| Mean GPU utilization (%) | `91.5938` |
| Peak GPU memory (MB) | `2904.1875` |
| Torch peak allocated (GB) | `2.1674` |
| Torch peak reserved (GB) | `2.2012` |

### Split-by-split evaluation

| Split | mAP50 | mAP50-95 | Precision | Recall | Inference ms/image |
| --- | --- | --- | --- | --- | --- |
| `train` | `0.2985` | `0.1758` | `0.4306` | `0.3283` | `53.5176` |
| `val` | `0.2893` | `0.1703` | `0.4107` | `0.3249` | `44.7413` |
| `test-dev` | `0.2394` | `0.1384` | `0.3637` | `0.2937` | `48.1247` |

## Next Planned Experiments

### 1. YOLO26x Fine-Tune

| Field | Value |
| --- | --- |
| Experiment name | `YOLO26x fine-tune` |
| Model / weights | `TBD` |
| Date | `TBD` |
| Notes | `teacher-quality training run placeholder` |

### Metrics

| Metric | Value |
| --- | --- |
| Val mAP50 | `TBD` |
| Val mAP50-95 | `TBD` |
| Val precision | `TBD` |
| Val recall | `TBD` |
| Median latency (ms/image) | `TBD` |
| Mean latency (ms/image) | `TBD` |
| Mean throughput (FPS) | `TBD` |
| Mean GPU utilization (%) | `TBD` |
| Peak GPU memory (MB) | `TBD` |

### 2. Supernet Search

| Field | Value |
| --- | --- |
| Experiment name | `Supernet search` |
| Search space / artifact | `TBD` |
| Date | `TBD` |
| Notes | `architecture search placeholder` |

### Metrics

| Metric | Value |
| --- | --- |
| Best candidate name | `TBD` |
| Best candidate val mAP50 | `TBD` |
| Best candidate val mAP50-95 | `TBD` |
| Best candidate median latency (ms/image) | `TBD` |
| Number of searched architectures | `TBD` |
| Pareto frontier summary | `TBD` |

### 3. Student Model

| Field | Value |
| --- | --- |
| Experiment name | `Student model` |
| Model / weights | `TBD` |
| Date | `TBD` |
| Notes | `KD or KD+NAS student placeholder` |

### Metrics

| Metric | Value |
| --- | --- |
| Val mAP50 | `TBD` |
| Val mAP50-95 | `TBD` |
| Val precision | `TBD` |
| Val recall | `TBD` |
| Median latency (ms/image) | `TBD` |
| Mean latency (ms/image) | `TBD` |
| Mean throughput (FPS) | `TBD` |
| Mean GPU utilization (%) | `TBD` |
| Peak GPU memory (MB) | `TBD` |

## Notes Template

Use this block when adding future experiments:

| Field | Value |
| --- | --- |
| Experiment name | `TBD` |
| Stage | `benchmark` / `fine-tune` / `supernet` / `student` |
| Model / artifact | `TBD` |
| Source report | `TBD` |
| Date | `TBD` |
| Key notes | `TBD` |
