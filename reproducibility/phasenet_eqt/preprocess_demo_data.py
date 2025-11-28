#!/usr/bin/env python3
"""
Preprocessing script for YAZEL RECOVAR demo.

This script categorizes PhaseNet picks based on catalog (truepicks/falsepicks),
processes them with RECOVAR, and saves metadata for fast demo loading.
"""

import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

from yazel_integration_sliding import (
    recovar_pick_cleaner_sliding,
    load_recovar_classifier
)
from demo_utils import load_example_picks


def preprocess_and_save(phasenet_pick_dir, catalog_path, model_path, output_dir):
    """
    Process all picks with RECOVAR and organize into truepicks/falsepicks directories.

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

    # Load all examples
    print("\nLoading all PhaseNet picks...")
    tp_examples, fp_examples = load_example_picks(phasenet_pick_dir, catalog)
    print(f"Found {len(tp_examples)} TRUE PICK examples")
    print(f"Found {len(fp_examples)} FALSE PICK examples")

    # Process and save TRUE PICKS
    print("\nProcessing TRUE PICKS with RECOVAR...")
    truepicks_metadata = []
    for example in tqdm(tp_examples, desc="TRUE PICKS"):
        result = recovar_pick_cleaner_sliding(example['stream'], classifier)

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
    for example in tqdm(fp_examples, desc="FALSE PICKS"):
        result = recovar_pick_cleaner_sliding(example['stream'], classifier)

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

    preprocess_and_save(PHASENET_PICK_DIR, CATALOG_PATH, MODEL_PATH, OUTPUT_DIR)
