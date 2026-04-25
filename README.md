# Dataset-Adaptive Retrieval for TS-RAG

> **Research Question:** Can a small, dataset-specific knowledge base match the performance of the original 4.77 GB global knowledge base in TS-RAG zero-shot time series forecasting?

## Overview

[TS-RAG](https://github.com/UConn-DSIS/TS-RAG) (NeurIPS 2025) is a retrieval-augmented generation framework for zero-shot time series forecasting. While effective, it requires a **4.77 GB global knowledge base** loaded into memory at inference time — making it impractical on resource-constrained hardware.

This project investigates whether a **dataset-specific, smaller knowledge base** built only from the target dataset's training split can match the forecasting performance of the original global database.

## Key Finding

| Setup | ETTh1 MSE | ETTh2 MSE | DB Size |
|---|---|---|---|
| Global DB (original) | 0.3556 | 0.2451 | 4,770 MB |
| Adaptive DB (ours) | 0.3556 | 0.2451 | 337 MB |

**96.5% memory reduction with zero performance loss.**

## Setup

```bash
conda create -n tsrag python=3.9 -y
conda activate tsrag
pip install faiss-cpu chronos-forecasting==1.5.1 numpy wandb scikit-learn gluonts pandas pyarrow matplotlib statsmodels
pip install transformers==4.40.0
```

## Usage

### Step 1: Build Adaptive Knowledge Bases
```bash
python3 build_adaptive_db.py
```

### Step 2: Run Evaluation

**Original global DB:**
```bash
bash script/zeroshot_chronos.sh
```

**Adaptive DB:**
```bash
bash script/zeroshot_adaptive.sh
```

## Results

| Dataset | MSE | MAE | RMSE | SMAPE | DB Setup |
|---|---|---|---|---|---|
| ETTh1 | 0.3556 | 0.3623 | 0.5963 | 67.60% | Global (4.77 GB) |
| ETTh2 | 0.2451 | 0.2981 | 0.4951 | 47.09% | Global (4.77 GB) |
| ETTh1 | 0.3556 | 0.3623 | 0.5963 | 67.60% | Adaptive (168 MB) |
| ETTh2 | 0.2451 | 0.2981 | 0.4951 | 47.09% | Adaptive (168 MB) |

## Based On

- **TS-RAG:** Ning et al., NeurIPS 2025. [arXiv:2503.07649](https://arxiv.org/abs/2503.07649)
- **ReTime:** Jing et al., CIKM 2022. [arXiv:2209.13525](https://arxiv.org/abs/2209.13525)
