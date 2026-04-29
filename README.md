# TS-RAG Adaptive Retrieval

**Can a 168 MB knowledge base replace a 4.77 GB one?**

This project extends [TS-RAG (Ning et al., NeurIPS 2025)](https://arxiv.org/abs/2503.07649) by introducing dataset-adaptive knowledge bases for time series forecasting with retrieval-augmented generation.

## Key Contribution

Instead of using a large global knowledge base (4.77 GB) containing mixed datasets, we propose using **dataset-specific adaptive knowledge bases** that only contain training data from the target dataset — reducing memory by 96.5% with no meaningful accuracy loss.

## Results

| Setup | Knowledge Base Size | ETTh1 MSE | ETTh1 MAE | ETTh2 MSE | ETTh2 MAE |
|-------|-------------------|-----------|-----------|-----------|-----------|
| Global DB (TS-RAG) | 4,770 MB | 0.3556 | 0.3623 | 0.2451 | 0.2981 |
| Adaptive DB (Ours) | **168 MB** | **0.3543** | 0.3643 | 0.2463 | 0.3005 |
| **Reduction** | **96.5%** | -0.0013 | +0.0020 | +0.0012 | +0.0024 |

**96.5% memory reduction with less than 0.5% performance difference.**

## What We Added

Two files on top of the original TS-RAG codebase:

- `build_adaptive_db.py` — builds dataset-specific knowledge bases from training split only (rows 0–8,640)
- `script/zeroshot_adaptive.sh` — evaluation script pointing to adaptive DB instead of global DB

## How to Run

### 1. Build Adaptive Knowledge Base
```bash
python build_adaptive_db.py
```
Outputs: `retrieval_database/adaptive/ETTh1_hour_512.pkl` (168 MB) and `ETTh2_hour_512.pkl` (168 MB)

### 2. Run Evaluation
```bash
bash script/zeroshot_adaptive.sh
```

## Requirements

Same as original TS-RAG. Key dependencies:
- `chronos-forecasting`
- `gluonts`
- `faiss-cpu`
- `transformers==4.40.0`

## Limitations

- Embeddings computed using mean pooling over Chronos-Bolt-Base encoder outputs (not original pipeline)
- Only tested on ETTh1 and ETTh2 datasets
- ARM module weights not fine-tuned on downstream task

## Authors

Tuana Turhan & Collin — CSC500, Quinnipiac University, 2026

## Based On

- TS-RAG: Ning et al., NeurIPS 2025 — [arXiv:2503.07649](https://arxiv.org/abs/2503.07649)
- ReTime: Jing et al., CIKM 2022 — [arXiv:2209.13525](https://arxiv.org/abs/2209.13525)
- Chronos-Bolt: Amazon, 2024
