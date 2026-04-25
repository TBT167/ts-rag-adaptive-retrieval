"""
Dataset-Adaptive Knowledge Base Builder for TS-RAG
Builds small, dataset-specific knowledge bases using only
the train split of each dataset, instead of the global 4.77 GB database.
"""

import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from chronos import ChronosPipeline

sys.path.insert(0, '.')
from retrieve import create_database, save_database

# Configuration
DATASETS_DIR = '../datasets/ETT-small'
OUTPUT_DIR = '../retrieval_database/adaptive'
LOOKBACK_LENGTH = 512
DEVICE = 'cpu'

# ETT train split borders (standard benchmark splits)
# Total ETTh1/ETTh2: 17420 rows
# Train: 0-8640, Val: 8640-11520, Test: 11520-17420
TRAIN_END = 8640

def build_adaptive_database(dataset_name, train_end=TRAIN_END):
    print(f'\n{"="*50}')
    print(f'Building adaptive database for {dataset_name}')
    print(f'Using only train split: rows 0 to {train_end}')
    print(f'{"="*50}')

    # Load dataset
    data_path = os.path.join(DATASETS_DIR, f'{dataset_name}.csv')
    df = pd.read_csv(data_path)
    
    # Use only train split
    df_train = df.iloc[:train_end]
    print(f'Full dataset size: {len(df)} rows')
    print(f'Train split size:  {len(df_train)} rows')

    # Load Chronos-Bolt embedding model
    print('Loading Chronos-Bolt model...')
    pipeline = ChronosPipeline.from_pretrained(
        '../checkpoints/chronos-bolt-base',
        device_map=DEVICE,
        torch_dtype=torch.float32,
    )

    # Build database for each variable
    variables = df_train.columns[1:]  # skip 'date' column
    databases = {}

    for var in variables:
        print(f'Processing variable: {var}')
        raw_data = df_train[var].tolist()
        timestamps = df_train['date'].tolist()
        metadata = {
            'dataset_name': dataset_name,
            'variable_name': var,
            'lookback_length': LOOKBACK_LENGTH,
            'frequency': 'hour',
        }
        database = create_database(raw_data, timestamps, LOOKBACK_LENGTH, pipeline, metadata)
        databases[var] = database
        print(f'  -> {len(raw_data) - LOOKBACK_LENGTH} embeddings created')

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_hour_{LOOKBACK_LENGTH}.pkl')
    save_database(databases, output_path)
    
    # Report size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\nSaved to: {output_path}')
    print(f'File size: {size_mb:.1f} MB')
    return output_path, size_mb

if __name__ == '__main__':
    results = {}
    
    for dataset in ['ETTh1', 'ETTh2']:
        path, size = build_adaptive_database(dataset)
        results[dataset] = {'path': path, 'size_mb': size}
    
    print('\n' + '='*50)
    print('SUMMARY')
    print('='*50)
    for dataset, info in results.items():
        print(f'{dataset}: {info["size_mb"]:.1f} MB -> {info["path"]}')
    print(f'\nOriginal global DB: ~4770 MB')
    print(f'Adaptive DBs combined: {sum(r["size_mb"] for r in results.values()):.1f} MB')
