import obspy
import pandas as pd
import numpy as np
from pathlib import Path
from yazel_integration import recovar_pick_cleaner_batch, load_recovar_classifier

MODEL_PATH = '/mnt/data_a/ege/recovar_models/exp_instance/representation_learning_autoencoder_ensemble/instance/split0/ep19.h5'

PHASENET_THRESHOLD = None

if PHASENET_THRESHOLD is None:
    phasenet_pick_dir = "filtered_phasenet_picks_dir"
else:
    phasenet_pick_dir = f"filtered_phasenet_picks_dir_thr_{PHASENET_THRESHOLD:.2f}"

phasenet_picks = pd.read_csv(f"{phasenet_pick_dir}/metadata.csv")

catalog_path = '/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/processed_catalogs/kara74a_phase_picks.csv'
catalog = pd.read_csv(catalog_path)
catalog['p_arrival_time'] = pd.to_datetime(catalog['p_arrival_time'])

p = Path(phasenet_pick_dir)

print("Loading classifier...")
classifier = load_recovar_classifier(MODEL_PATH)

print("Loading all waveforms...")
files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix == '.mseed'])
streams = []
valid_files = []

for file in files:
    try:
        stream = obspy.read(file)
        stream.merge()
        streams.append(stream)
        valid_files.append(file)
    except Exception as e:
        print(f"Error loading {file.name}: {e}")
        continue

print(f"Loaded {len(streams)} waveforms")

print("Processing in batches of 256...")
results = recovar_pick_cleaner_batch(streams=streams, classifier=classifier, batch_size=256)

print("Creating comparison data...")
comparison_data = []

for idx, (file, result) in enumerate(zip(valid_files, results)):
    stream = streams[idx]
    station = stream[0].stats.station

    pick_row = phasenet_picks[phasenet_picks['filename'] == file.name]

    if pick_row.empty:
        continue

    phasenet_pick = pd.to_datetime(pick_row['pick_time'].values[0], format='mixed')
    window_start = pd.to_datetime(pick_row['start_time'].values[0], format='mixed')
    window_end = pd.to_datetime(pick_row['end_time'].values[0], format='mixed')

    catalog_picks = catalog[
        (catalog['station'] == station) &
        (catalog['p_arrival_time'] >= window_start) &
        (catalog['p_arrival_time'] <= window_end)
    ]

    catalog_pick = catalog_picks.iloc[0]['p_arrival_time'] if not catalog_picks.empty else None

    has_catalog = catalog_pick is not None
    has_phasenet = phasenet_pick is not None

    if has_catalog and has_phasenet:
        status = "Both"
    elif has_catalog:
        status = "Catalog only"
    elif has_phasenet:
        status = "PhaseNet only"
    else:
        status = "None"

    comparison_data.append({
        'filename': file.name,
        'station': station,
        'catalog_pick': catalog_pick,
        'phasenet_pick': phasenet_pick,
        'model_score': result,
        'detection_status': status
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n=== DETECTION STATISTICS ===\n")
print(f"Total windows: {len(comparison_df)}")

both = comparison_df[comparison_df['detection_status'] == 'Both']['model_score'].values
phasenet_only = comparison_df[comparison_df['detection_status'] == 'PhaseNet only']['model_score'].values

print(f"Both (catalog + PhaseNet): {len(both)}")
print(f"PhaseNet only: {len(phasenet_only)}")
print(f"Catalog only: {len(comparison_df[comparison_df['detection_status'] == 'Catalog only'])}\n")

print("=== MODEL SCORES: BOTH (TRUE POSITIVES) ===")
print(f"Count: {len(both)}")
print(f"Mean: {np.mean(both):.3f}")
print(f"Std: {np.std(both):.3f}")
print(f"Min: {np.min(both):.3f}")
print(f"Max: {np.max(both):.3f}")

print("\n=== MODEL SCORES: PHASENET ONLY (FALSE POSITIVES) ===")
print(f"Count: {len(phasenet_only)}")
print(f"Mean: {np.mean(phasenet_only):.3f}")
print(f"Std: {np.std(phasenet_only):.3f}")
print(f"Min: {np.min(phasenet_only):.3f}")
print(f"Max: {np.max(phasenet_only):.3f}")

from sklearn.metrics import f1_score

def find_best_f1_threshold(scores, labels, num_thresholds=500):
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)

    best_f1 = -1.0
    best_thr = thresholds[0]

    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = t

    return best_thr, best_f1

print("\n=== FINDING BEST THRESHOLD ===")
all_scores = comparison_df["model_score"].values
all_labels = ~comparison_df["catalog_pick"].isna()

best_threshold, best_f1 = find_best_f1_threshold(all_scores, all_labels)
print(f"Best threshold: {best_threshold:.6f}")
print(f"Best F1 score: {best_f1:.3f}")

comparison_df['recovar_decision'] = comparison_df['model_score'] >= best_threshold

tp = np.sum((comparison_df['detection_status'] == 'Both') & comparison_df['recovar_decision'])
fp = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & comparison_df['recovar_decision'])
fn = np.sum((comparison_df['detection_status'] == 'Both') & ~comparison_df['recovar_decision'])
tn = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & ~comparison_df['recovar_decision'])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n=== FINAL PERFORMANCE WITH THRESHOLD {best_threshold:.6f} ===")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Evaluate manual thresholds
manual_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\n=== PERFORMANCE AT MANUAL THRESHOLDS ===")
for manual_thr in manual_thresholds:
    comparison_df['manual_decision'] = comparison_df['model_score'] >= manual_thr

    tp_m = np.sum((comparison_df['detection_status'] == 'Both') & comparison_df['manual_decision'])
    fp_m = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & comparison_df['manual_decision'])
    fn_m = np.sum((comparison_df['detection_status'] == 'Both') & ~comparison_df['manual_decision'])
    tn_m = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & ~comparison_df['manual_decision'])

    precision_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0
    recall_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
    f1_m = 2 * precision_m * recall_m / (precision_m + recall_m) if (precision_m + recall_m) > 0 else 0

    print(f"\nThreshold: {manual_thr:.2f}")
    print(f"  TP={tp_m}, FP={fp_m}, FN={fn_m}, TN={tn_m}")
    print(f"  Precision={precision_m:.3f}, Recall={recall_m:.3f}, F1={f1_m:.3f}")

comparison_df.to_csv('SLVT_pick_comparison_batched.csv', index=False)
print(f"\nResults saved to: SLVT_pick_comparison_batched.csv")
