#!/usr/bin/env python3
"""
Preprocessing script for YAZEL RECOVAR demo.

This script categorizes PhaseNet picks based on catalog (truepicks/falsepicks),
processes them with RECOVAR, and saves metadata for fast demo loading.
"""

import pandas as pd
import shutil
from pathlib import Path

from yazel_integration_sliding import (
    recovar_pick_cleaner_sliding_batch,
    load_recovar_classifier
)
from demo_utils import load_example_picks


def preprocess_and_save(phasenet_pick_dir, catalog_path, model_path, output_dir, max_samples=None):
    """
    Process picks with RECOVAR and organize into truepicks/falsepicks directories.

    Parameters
    ----------
    phasenet_pick_dir : str
        Directory containing PhaseNet pick files
    catalog_path : str
        Path to catalog CSV file
    model_path : str
        Path to RECOVAR model
    output_dir : str
        Directory to save preprocessed data
    max_samples : int or None, optional
        Maximum number of samples to process (for demo). If None, process all.
    """
    output_path = Path(output_dir)
    truepicks_dir = output_path / 'truepicks'
    falsepicks_dir = output_path / 'falsepicks'

    # Create directories
    truepicks_dir.mkdir(exist_ok=True, parents=True)
    falsepicks_dir.mkdir(exist_ok=True, parents=True)

    # Load catalog
    print("Loading catalog...")
    catalog = pd.read_csv(catalog_path)
    catalog['p_arrival_time'] = pd.to_datetime(catalog['p_arrival_time'])
    catalog = catalog[catalog['station'] == 'SLVT']
    print(f"Loaded {len(catalog)} catalog picks for station SLVT")

    # Load RECOVAR classifier
    print("\nLoading RECOVAR classifier...")
    classifier = load_recovar_classifier(model_path)
    print("Classifier loaded successfully!")

    # Load examples
    print("\nLoading PhaseNet picks...")
    # Set min_tp/min_fp high so it doesn't stop early
    min_examples = max_samples if max_samples is not None else 999999
    tp_examples, fp_examples = load_example_picks(
        phasenet_pick_dir, catalog,
        max_files=max_samples,
        min_tp=min_examples,
        min_fp=min_examples
    )

    # Limit samples if specified
    if max_samples is not None:
        tp_examples = tp_examples[:max_samples]
        fp_examples = fp_examples[:max_samples]

    print(f"Found {len(tp_examples)} TRUE PICK examples")
    print(f"Found {len(fp_examples)} FALSE PICK examples")

    # Process and save TRUE PICKS
    print("\nProcessing TRUE PICKS with RECOVAR...")
    truepicks_metadata = []
    batch_size = 32

    for batch_start in range(0, len(tp_examples), batch_size):
        batch_end = min(batch_start + batch_size, len(tp_examples))
        batch = tp_examples[batch_start:batch_end]

        print(f"  Processing TRUE PICKS {batch_start+1}-{batch_end}/{len(tp_examples)}...")

        # Batch process
        streams = [ex['stream'] for ex in batch]
        results = recovar_pick_cleaner_sliding_batch(streams, classifier)

        # Save results
        for example, result in zip(batch, results):
            # Copy mseed file
            dest = truepicks_dir / example['file'].name
            shutil.copy2(example['file'], dest)

            # Save metadata
            truepicks_metadata.append({
                'filename': example['file'].name,
                'station': example['station'],
                'phasenet_pick': example['phasenet_pick'],
                'catalog_pick': example['catalog_pick'],
                'window_start': example['window_start'],
                'window_end': example['window_end'],
                'mean_score': result['mean_score'],
                'max_score': result['max_score']
            })

    # Process and save FALSE PICKS
    print("\nProcessing FALSE PICKS with RECOVAR...")
    falsepicks_metadata = []

    for batch_start in range(0, len(fp_examples), batch_size):
        batch_end = min(batch_start + batch_size, len(fp_examples))
        batch = fp_examples[batch_start:batch_end]

        print(f"  Processing FALSE PICKS {batch_start+1}-{batch_end}/{len(fp_examples)}...")

        # Batch process
        streams = [ex['stream'] for ex in batch]
        results = recovar_pick_cleaner_sliding_batch(streams, classifier)

        # Save results
        for example, result in zip(batch, results):
            # Copy mseed file
            dest = falsepicks_dir / example['file'].name
            shutil.copy2(example['file'], dest)

            # Save metadata
            falsepicks_metadata.append({
                'filename': example['file'].name,
                'station': example['station'],
                'phasenet_pick': example['phasenet_pick'],
                'catalog_pick': example['catalog_pick'],
                'window_start': example['window_start'],
                'window_end': example['window_end'],
                'mean_score': result['mean_score'],
                'max_score': result['max_score']
            })

    # Save metadata CSVs
    pd.DataFrame(truepicks_metadata).to_csv(output_path / 'truepicks_metadata.csv', index=False)
    pd.DataFrame(falsepicks_metadata).to_csv(output_path / 'falsepicks_metadata.csv', index=False)

    print(f"\nPreprocessed data saved to: {output_path}")
    print(f"  - {len(tp_examples)} TRUE PICK examples in {truepicks_dir}")
    print(f"  - {len(fp_examples)} FALSE PICK examples in {falsepicks_dir}")
    print(f"  - Metadata saved with RECOVAR scores")


if __name__ == '__main__':
    # Configuration
    MODEL_PATH = '/mnt/data_a/ege/recovar_models/exp_instance/representation_learning_autoencoder_ensemble/instance/split0/ep19.h5'
    PHASENET_THRESHOLD = 0.32
    PHASENET_PICK_DIR = f"filtered_phasenet_picks_dir_thr_{PHASENET_THRESHOLD:.2f}"
    CATALOG_PATH = '/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/processed_catalogs/kara74a_phase_picks.csv'
    OUTPUT_DIR = 'preprocessed_demo_data'
    MAX_SAMPLES = 200  # Limit for demo - set to None to process all

    preprocess_and_save(PHASENET_PICK_DIR, CATALOG_PATH, MODEL_PATH, OUTPUT_DIR, max_samples=MAX_SAMPLES)
