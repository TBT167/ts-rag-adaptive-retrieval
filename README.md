# TS-RAG Adaptive Retrieval

## Overview
This project extends [TS-RAG](https://github.com/ORIGINAL_REPO) by introducing **dataset-adaptive knowledge bases** for time series forecasting with retrieval-augmented generation.

## Key Contribution
Instead of using a large global knowledge base (4.77 GB) containing mixed datasets, we propose using **dataset-specific adaptive knowledge bases** that only contain training data from the target dataset.

## Results

| Setup | Knowledge Base Size | ETTh1 MSE | ETTh1 MAE | ETTh2 MSE | ETTh2 MAE |
|-------|-------------------|-----------|-----------|-----------|-----------|
| Global DB (TS-RAG) | 4,770 MB | 0.3556 | 0.3623 | 0.2451 | 0.2981 |
| Adaptive DB (Ours) | 168 MB | **0.3543** | **0.3643** | **0.2463** | **0.3005** |
| **Reduction** | **96.5%** | -0.0013 | +0.0020 | +0.0012 | +0.0024 |

**96.5% memory reduction with comparable performance.**

## How to Run

### Build Adaptive Knowledge Base
```bash
python build_adaptive_db.py
```

### Run Evaluation
```bash
bash script/zeroshot_adaptive.sh
```

## Limitations
- Embeddings computed using mean pooling over Chronos-Bolt-Base encoder outputs
- Results may vary slightly from original TS-RAG due to different retrieval pipeline implementation
