# VisDrone-KD-NAS: Distillation-Aware Student Architecture Search for Efficient UAV Object Detection

**Status: Active — working document**

---

## 1. One-Line Summary

Search for the optimal lightweight YOLO26 student architecture under a strong YOLO26x teacher using knowledge distillation as the search objective, targeting VisDrone small-object detection at edge-deployable latency.

---

## 2. Motivation & Gap

### The problem
State-of-the-art models on VisDrone (Drone-DETR 53.9% mAP@0.5, UAV-DETR 51.6%) are transformer-based and too heavy for edge deployment. Lightweight YOLO-family models lag 15–20 mAP points behind. Knowledge distillation is the natural bridge — but existing KD work uses a **fixed student architecture**. The question nobody has asked:

> *Does the choice of student architecture interact with the teacher's knowledge? Can we search for the architecture that best absorbs the teacher's small-object detection capability?*

### Why YOLO26
YOLO26 (Ultralytics, Sep 2025 / arXiv 2606.03748) is the newest and strongest YOLO family:
- **COCO mAP:** 40.9–57.5% across n/s/m/l/x at 1.7–11.8ms T4 latency
- **STAL** (Small-Target-Aware Label Assignment) — built-in small-object focus, directly relevant to VisDrone
- **NMS-free** end-to-end inference, **ProgLoss**, **MuSGD** optimizer
- All sizes (n/s/m/l/x) share the same Ultralytics codebase — ideal for a supernet

### The corpus gap
- **0 / 552 papers** in the merged knowledge graph combine NAS + KD for object detection.
- UAV-targeted NAS only appears in 2025–2026 (`Deployment-Aware NAS`, `AutoUAVFormer`) — both search whole detectors with heavy compute and no KD.
- KD for detection is well-explored (48 papers) but always with fixed architectures.
- No published VisDrone evaluation for YOLO26 yet — this study provides it as a by-product.

---

## 3. Proposed Method

```
YOLO26x (teacher, COCO pretrained → fine-tuned on VisDrone)
    │
    │  offline inference → pseudo-labels (soft class scores + boxes)
    │  + online feature distillation at P3/P4/P5
    ▼
NAS Supernet (YOLO26 family search space)
    │  search objective: mAP_val + λ_kd · L_kd + λ_lat · latency_penalty
    │  NSGA-II multi-objective: maximize mAP, minimize latency
    ▼
Top-K student architectures (Pareto front: accuracy vs. latency)
    │  retrain each with full KD from teacher
    ▼
Deployed lightweight detectors, VisDrone-optimized
```

### KD strategy: response + feature (same family enables both)

Using YOLO26x as teacher (same Ultralytics family as student) unlocks two levels of distillation:

| Level | What is distilled | Loss |
|-------|------------------|------|
| **Response** | Teacher detection outputs (cls logits, box regression) | KL divergence + L1 |
| **Feature** | Teacher FPN feature maps at P3/P4/P5 | L2 after 1×1 adapter |

Feature-level KD is feasible because teacher and student share the same YOLO26 backbone blocks and FPN structure — P3/P4/P5 feature dimensions are directly compatible. STAL in both teacher and student means the distilled small-object assignment signal is architecturally aligned.

### KD fairness rule
One KD recipe, one set of λ's, tuned once on YOLO26n and then **frozen for every student**. Architecture is the only variable. This is the core internal validity requirement for the NAS comparison.

### Search space
Variants of the YOLO26 backbone and neck along three axes:

| Axis | Options |
|------|---------|
| Width multiplier | 0.25 / 0.375 / 0.50 / 0.625 |
| Depth multiplier | 0.33 / 0.44 / 0.67 |
| Neck ops (per edge) | YOLO26 default / Ghost variant / CSP-small / skip |

Approximately ~200–400 discrete architectures in the space (enumerable for NSGA-II evaluation).

---

## 4. Datasets & Baselines

| Role | Choice |
|------|--------|
| Primary train/search | VisDrone-DET (6,471 train / 548 val) |
| Generalization check | UAVDT (optional, if compute allows — same aerial domain, different scenes) |
| Teacher pre-training | COCO → fine-tune on VisDrone |
| Student search + retrain | VisDrone only |

### Baselines to beat

| Baseline | mAP@0.5 | Params | Source |
|----------|---------|--------|--------|
| YOLO26n (no KD, no NAS) | TBD on VisDrone | ~3M | Ultralytics |
| YOLO26n + KD only (fixed arch) | TBD | ~3M | our impl |
| RemDet-Tiny (AAAI 2025) | SOTA efficient | — | [github](https://github.com/hzai-zjnu/remdet) |
| Drone-DETR (heavy SOTA) | 53.9% | 28.7M | [github](https://github.com/Ame1999c/Drone-DETR) |
| UAV-DETR | 51.6% | 16.8M | [github](https://github.com/ValiantDiligent/UAV-DETR) |
| YOLO26x teacher (upper bound) | TBD on VisDrone | ~57M | Ultralytics |

**Key comparison (the headline experiment):** teacher-aware NAS vs. teacher-blind NAS at matched latency (±10%). Same supernet, same NSGA-II, only the search objective differs.

### Metrics
- mAP@0.5, mAP-small (spotlight metric for VisDrone)
- Measured T4 latency: TensorRT FP16, batch 1, median of 200 runs after 50 warmup
- Params, GFLOPs

---

## 5. Kaggle/Colab Training Plan

**Total: ~45–55 GPU-hours ≈ 2 Kaggle weeks.**

| Phase | Details | Est. time |
|-------|---------|-----------|
| Teacher fine-tune | YOLO26x, COCO pretrained → VisDrone, 100 epochs, 640px | 5–6 h |
| Pseudo-label generation | Run teacher on all train images, save soft labels | 1 h |
| Supernet training | YOLO26 supernet + KD loss, VisDrone, 100 epochs, 512px | 15–20 h |
| NSGA-II search | Evaluate ~200 archs on val, 2 objectives (mAP + latency) | 3–5 h |
| Retrain top-5 students | 640px, 150 epochs, full KD | 4 h each → 20 h |
| Baseline retrains | YOLO26n no-KD, YOLO26n KD-only | 4 h |

**Checkpoint-resume required:** every phase saves checkpoints to Kaggle Dataset / Google Drive. Sessions die at 12 h (Kaggle) / 4 h (Colab).

---

## 6. Research Questions & Experiment Map

| RQ | Question | Experiment |
|----|---------|------------|
| RQ1 | Does KD-aware NAS find better students than KD-blind NAS? | E1: same supernet, compare search objectives |
| RQ2 | What is the accuracy-latency Pareto frontier for KD-searched YOLO26 on VisDrone? | E2: plot top-5 students vs. baselines |
| RQ3 | Does the searched student generalize to other UAV datasets? | E3: transfer table on UAVDT (optional) |
| RQ4 | Which architectural axes (width / depth / neck ops) matter most under KD? | E4: ablation over search space axes |

---

## 7. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Supernet ranking inconsistency on 7k images | High | Pre-train supernet on COCO before VisDrone fine-tune |
| YOLO26 VisDrone performance unconfirmed | Medium | Verify teacher mAP early (Phase 1); pivot to YOLOv11x if YOLO26 underperforms |
| KD signal weak at lightweight student capacity | Medium | Report teacher-student gap honestly; frame as upper-bound study |
| RemDet already achieves good efficiency | Medium | Differentiate on deployment story (latency Pareto family) and NAS+KD novelty |
| Weight-sharing ranking inconsistency (known NAS problem) | Medium | Validate top-5 by retraining; report rank correlation |
| Kaggle quota exhausted mid-run | Low | Chunk into 10 h sessions; all phases are checkpoint-resumable |

---

## 8. Expected Contributions

1. First study combining NAS + KD for UAV/small-object detection (fills 0/552 corpus gap).
2. First VisDrone evaluation of YOLO26, establishing it as a strong UAV detection baseline.
3. Evidence that teacher-aware student search outperforms teacher-blind search at matched latency.
4. VisDrone accuracy-latency Pareto family of lightweight YOLO26 detectors with public weights.
5. Frozen KD recipe + NSGA-II search protocol reproducible on free-tier GPUs.

**Target venues:** AAAI, WACV, ICRA/IROS (deployment angle), *Drones* journal, or CVPR efficient-vision workshops.

---

## 9. Key References

- `NAS-FPN` — neck search prior work
- `TF-NAS` (2020) — latency-constrained differentiable NAS
- `HardCoRe-NAS` (2021) — hard constraint NAS
- `Prior-Guided One-shot NAS` (2022) — supernet ranking criticism
- `RemDet` (AAAI 2025) — lightweight UAV detection SOTA baseline
- `Deployment-Aware NAS for Lightweight UAV Object Detectors` (2026) — adjacent UAV NAS
- Drone-DETR (2024) — VisDrone DETR-based upper bound
- UAV-DETR (2025) — lighter DETR baseline
- YOLO26 / Ultralytics (arXiv 2606.03748) — teacher + student skeleton

---

_Working document — 2026-06-12_
