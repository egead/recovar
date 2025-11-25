#!/usr/bin/env python3
"""
Diagnostic script to investigate why RECOVAR classifier cannot distinguish
between true earthquake picks and false picks.
"""

import numpy as np
import obspy
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import functions from yazel_integration
from yazel_integration import preprocess, load_recovar_classifier

def diagnose_window_centering(window_dir, metadata_path, catalog_path, num_samples=10):
    """
    Check if picks are actually centered in the extracted windows.

    Args:
        window_dir: Path to directory with .mseed windows
        metadata_path: Path to metadata CSV with pick times
        catalog_path: Path to catalog CSV
        num_samples: Number of windows to examine
    """
    print("=" * 80)
    print("DIAGNOSTIC 1: PICK CENTERING ANALYSIS")
    print("=" * 80)

    metadata = pd.read_csv(metadata_path)
    catalog = pd.read_csv(catalog_path)
    catalog['p_arrival_time'] = pd.to_datetime(catalog['p_arrival_time'])

    p = Path(window_dir)
    files = sorted([f for f in p.iterdir() if f.suffix == '.mseed'])

    centering_errors = []

    for i, file in enumerate(files[:num_samples]):
        print(f"\n--- Window {i+1}/{num_samples}: {file.name} ---")

        stream = obspy.read(str(file))
        stream.merge()

        # Get metadata for this window
        pick_row = metadata[metadata['filename'] == file.name]
        if pick_row.empty:
            print(f"WARNING: No metadata found for {file.name}")
            continue

        phasenet_pick = pd.to_datetime(pick_row['pick_time'].values[0], format='mixed')
        window_start = obspy.UTCDateTime(pick_row['start_time'].values[0])
        window_end = obspy.UTCDateTime(pick_row['end_time'].values[0])

        # Calculate window center
        window_duration = window_end - window_start
        window_center = window_start + (window_duration / 2.0)

        # Check if there's a catalog pick
        station = stream[0].stats.station
        catalog_picks = catalog[
            (catalog['station'] == station) &
            (catalog['p_arrival_time'] >= pd.to_datetime(window_start.isoformat())) &
            (catalog['p_arrival_time'] <= pd.to_datetime(window_end.isoformat()))
        ]

        has_catalog = len(catalog_picks) > 0
        status = "TRUE POSITIVE" if has_catalog else "FALSE POSITIVE"

        print(f"Status: {status}")
        print(f"Station: {station}")
        print(f"Window start: {window_start}")
        print(f"Window end: {window_end}")
        print(f"Window duration: {window_duration:.2f} seconds")
        print(f"Window center: {window_center}")
        print(f"PhaseNet pick: {obspy.UTCDateTime(phasenet_pick.isoformat())}")

        phasenet_offset = obspy.UTCDateTime(phasenet_pick.isoformat()) - window_center
        print(f"PhaseNet pick offset from center: {phasenet_offset:.3f} seconds")

        if has_catalog:
            catalog_pick_time = catalog_picks.iloc[0]['p_arrival_time']
            catalog_offset = obspy.UTCDateTime(catalog_pick_time.isoformat()) - window_center
            print(f"Catalog pick: {catalog_pick_time}")
            print(f"Catalog pick offset from center: {catalog_offset:.3f} seconds")
            centering_errors.append(abs(catalog_offset))
        else:
            centering_errors.append(abs(phasenet_offset))

        # Check window dimensions
        print(f"Window shape: {len(stream[0].data)} samples")
        print(f"Sampling rate: {stream[0].stats.sampling_rate} Hz")
        expected_samples = int(window_duration * stream[0].stats.sampling_rate)
        print(f"Expected samples: {expected_samples}")

        # Check what the middle 3000 samples would be
        middle_start = len(stream[0].data) // 2 - 1500
        middle_end = len(stream[0].data) // 2 + 1500
        print(f"Middle 3000 samples: [{middle_start}:{middle_end}]")

    if centering_errors:
        print(f"\n\nCENTERING ERROR STATISTICS:")
        print(f"Mean absolute offset: {np.mean(centering_errors):.3f} seconds")
        print(f"Std absolute offset: {np.std(centering_errors):.3f} seconds")
        print(f"Max absolute offset: {np.max(centering_errors):.3f} seconds")
        print(f"Min absolute offset: {np.min(centering_errors):.3f} seconds")


def diagnose_preprocessing(window_dir, metadata_path, catalog_path, num_samples=5):
    """
    Check preprocessing and visualize the data that goes into the model.
    """
    print("\n\n" + "=" * 80)
    print("DIAGNOSTIC 2: PREPROCESSING ANALYSIS")
    print("=" * 80)

    metadata = pd.read_csv(metadata_path)
    catalog = pd.read_csv(catalog_path)
    catalog['p_arrival_time'] = pd.to_datetime(catalog['p_arrival_time'])

    p = Path(window_dir)
    files = sorted([f for f in p.iterdir() if f.suffix == '.mseed'])

    # Create output directory for plots
    output_dir = Path(window_dir).parent / "diagnostic_plots"
    output_dir.mkdir(exist_ok=True)

    for i, file in enumerate(files[:num_samples]):
        print(f"\n--- Preprocessing window {i+1}/{num_samples}: {file.name} ---")

        stream = obspy.read(str(file))
        stream.merge()
        stream = stream.select(channel="HH*")

        # Get metadata
        pick_row = metadata[metadata['filename'] == file.name]
        if pick_row.empty:
            continue

        station = stream[0].stats.station
        window_start = pd.to_datetime(pick_row['start_time'].values[0], format='mixed')
        window_end = pd.to_datetime(pick_row['end_time'].values[0], format='mixed')

        catalog_picks = catalog[
            (catalog['station'] == station) &
            (catalog['p_arrival_time'] >= window_start) &
            (catalog['p_arrival_time'] <= window_end)
        ]

        has_catalog = len(catalog_picks) > 0
        status = "TRUE_POS" if has_catalog else "FALSE_POS"

        # Create figure with 3 components
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        channels = ['HHE', 'HHN', 'HHZ']
        for ch_idx, channel in enumerate(channels):
            tr = stream.select(channel=channel)[0]

            # Raw data
            times = tr.times()
            axes[ch_idx, 0].plot(times, tr.data, 'k-', linewidth=0.5)
            axes[ch_idx, 0].set_ylabel(channel)
            axes[ch_idx, 0].set_title(f'{channel} - Raw')
            axes[ch_idx, 0].grid(True, alpha=0.3)

            # Mark center
            center_sample = len(tr.data) // 2
            center_time = times[center_sample]
            axes[ch_idx, 0].axvline(center_time, color='red', linestyle='--',
                                     label='Window center', alpha=0.7)

            # Mark the region that will be extracted (middle 3000 samples)
            middle_start_sample = len(tr.data) // 2 - 1500
            middle_end_sample = len(tr.data) // 2 + 1500
            middle_start_time = times[middle_start_sample]
            middle_end_time = times[middle_end_sample]
            axes[ch_idx, 0].axvspan(middle_start_time, middle_end_time,
                                     alpha=0.2, color='blue',
                                     label='Extracted region (3000 samples)')

            # Processed data
            processed = preprocess(tr.data, sampling_rate=tr.stats.sampling_rate)
            axes[ch_idx, 1].plot(times, processed, 'b-', linewidth=0.5)
            axes[ch_idx, 1].set_ylabel(channel)
            axes[ch_idx, 1].set_title(f'{channel} - Preprocessed (1-20 Hz)')
            axes[ch_idx, 1].grid(True, alpha=0.3)
            axes[ch_idx, 1].axvline(center_time, color='red', linestyle='--',
                                     alpha=0.7)
            axes[ch_idx, 1].axvspan(middle_start_time, middle_end_time,
                                     alpha=0.2, color='blue')

            # Check for NaN or inf
            if np.isnan(processed).any():
                print(f"WARNING: NaN values in processed {channel}")
            if np.isinf(processed).any():
                print(f"WARNING: Inf values in processed {channel}")

            # Statistics
            print(f"{channel} - Raw: mean={np.mean(tr.data):.2e}, std={np.std(tr.data):.2e}, "
                  f"min={np.min(tr.data):.2e}, max={np.max(tr.data):.2e}")
            print(f"{channel} - Processed: mean={np.mean(processed):.2e}, std={np.std(processed):.2e}, "
                  f"min={np.min(processed):.2e}, max={np.max(processed):.2e}")

        axes[0, 0].legend(loc='upper right', fontsize=8)
        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')

        fig.suptitle(f'{file.name} - {status}', fontsize=12, fontweight='bold')

        output_file = output_dir / f"diagnostic_{i:03d}_{status}_{file.stem}.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_file}")


def diagnose_model_input(window_dir, metadata_path, catalog_path, model_path, num_samples=5):
    """
    Examine the exact data that goes into the model and the output scores.
    """
    print("\n\n" + "=" * 80)
    print("DIAGNOSTIC 3: MODEL INPUT/OUTPUT ANALYSIS")
    print("=" * 80)

    metadata = pd.read_csv(metadata_path)
    catalog = pd.read_csv(catalog_path)
    catalog['p_arrival_time'] = pd.to_datetime(catalog['p_arrival_time'])

    # Load model
    print("\nLoading RECOVAR classifier...")
    classifier = load_recovar_classifier(model_path)
    print("Model loaded successfully")

    p = Path(window_dir)
    files = sorted([f for f in p.iterdir() if f.suffix == '.mseed'])

    output_dir = Path(window_dir).parent / "diagnostic_plots"
    output_dir.mkdir(exist_ok=True)

    true_pos_scores = []
    false_pos_scores = []

    for i, file in enumerate(files[:num_samples]):
        print(f"\n--- Model input {i+1}/{num_samples}: {file.name} ---")

        stream = obspy.read(str(file))
        stream.merge()

        # Get metadata
        pick_row = metadata[metadata['filename'] == file.name]
        if pick_row.empty:
            continue

        station = stream[0].stats.station
        window_start = pd.to_datetime(pick_row['start_time'].values[0], format='mixed')
        window_end = pd.to_datetime(pick_row['end_time'].values[0], format='mixed')

        catalog_picks = catalog[
            (catalog['station'] == station) &
            (catalog['p_arrival_time'] >= window_start) &
            (catalog['p_arrival_time'] <= window_end)
        ]

        has_catalog = len(catalog_picks) > 0
        status = "TRUE_POS" if has_catalog else "FALSE_POS"

        # Preprocess each channel
        z_trace = stream.select(channel="HHZ")[0]
        n_trace = stream.select(channel="HHN")[0]
        e_trace = stream.select(channel="HHE")[0]

        e_processed = preprocess(e_trace.data)
        n_processed = preprocess(n_trace.data)
        z_processed = preprocess(z_trace.data)

        print(f"Original window length: {len(e_processed)} samples")

        # Extract middle 3000 samples
        e_extracted = e_processed[len(e_processed) // 2 - 1500 : len(e_processed) // 2 + 1500]
        n_extracted = n_processed[len(n_processed) // 2 - 1500 : len(n_processed) // 2 + 1500]
        z_extracted = z_processed[len(z_processed) // 2 - 1500 : len(z_processed) // 2 + 1500]

        print(f"Extracted window length: {len(e_extracted)} samples")
        print(f"Expected: 3000 samples")

        if len(e_extracted) != 3000:
            print(f"ERROR: Extracted window has {len(e_extracted)} samples, expected 3000!")

        # Stack into model input format
        waveform = np.stack([e_extracted, n_extracted, z_extracted], axis=-1)
        waveform_batch = np.expand_dims(waveform, axis=0)

        print(f"Model input shape: {waveform_batch.shape}")
        print(f"Expected shape: (1, 3000, 3)")

        # Run model
        score = classifier(waveform_batch)
        print(f"Model score: {score[0]:.6f}")
        print(f"Status: {status}")

        if has_catalog:
            true_pos_scores.append(score[0])
        else:
            false_pos_scores.append(score[0])

        # Visualize the extracted window that goes into the model
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        time_axis = np.arange(3000) / 100.0  # 100 Hz sampling

        axes[0].plot(time_axis, e_extracted, 'k-', linewidth=0.5)
        axes[0].set_ylabel('HHE')
        axes[0].set_title('East Component (Input to Model)')
        axes[0].axvline(15.0, color='red', linestyle='--', label='Center (expected pick location)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(time_axis, n_extracted, 'k-', linewidth=0.5)
        axes[1].set_ylabel('HHN')
        axes[1].set_title('North Component (Input to Model)')
        axes[1].axvline(15.0, color='red', linestyle='--')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(time_axis, z_extracted, 'k-', linewidth=0.5)
        axes[2].set_ylabel('HHZ')
        axes[2].set_title('Vertical Component (Input to Model)')
        axes[2].axvline(15.0, color='red', linestyle='--')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel('Time (s)')

        fig.suptitle(f'{file.name} - {status} - Score: {score[0]:.6f}',
                     fontsize=12, fontweight='bold')

        output_file = output_dir / f"model_input_{i:03d}_{status}_{file.stem}.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_file}")

    if true_pos_scores and false_pos_scores:
        print(f"\n\nMODEL SCORE COMPARISON:")
        print(f"True Positives (n={len(true_pos_scores)}):")
        print(f"  Mean: {np.mean(true_pos_scores):.6f}")
        print(f"  Std: {np.std(true_pos_scores):.6f}")
        print(f"  Range: [{np.min(true_pos_scores):.6f}, {np.max(true_pos_scores):.6f}]")
        print(f"\nFalse Positives (n={len(false_pos_scores)}):")
        print(f"  Mean: {np.mean(false_pos_scores):.6f}")
        print(f"  Std: {np.std(false_pos_scores):.6f}")
        print(f"  Range: [{np.min(false_pos_scores):.6f}, {np.max(false_pos_scores):.6f}]")


def main():
    """Run all diagnostics."""

    # Configuration
    window_dir = "/home/ege/recovar/reproducibility/phasenet_eqt/phasenet_windows_SLVT"
    metadata_path = "/home/ege/recovar/reproducibility/phasenet_eqt/phasenet_windows_SLVT/metadata.csv"
    catalog_path = "/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/processed_catalogs/kara74a_phase_picks.csv"
    model_path = "/mnt/data_a/ege/recovar_models/exp_instance/representation_learning_autoencoder_ensemble/instance/split0/ep19.h5"

    print("RECOVAR CLASSIFIER DIAGNOSTIC TOOL")
    print("=" * 80)
    print(f"Window directory: {window_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Catalog: {catalog_path}")
    print(f"Model: {model_path}")
    print("=" * 80)

    # Run diagnostics
    diagnose_window_centering(window_dir, metadata_path, catalog_path, num_samples=10)
    diagnose_preprocessing(window_dir, metadata_path, catalog_path, num_samples=10)
    diagnose_model_input(window_dir, metadata_path, catalog_path, model_path, num_samples=10)

    print("\n\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("Check the 'diagnostic_plots' directory for visualizations")


if __name__ == "__main__":
    main()
