import obspy
import pandas as pd
import numpy as np
from pathlib import Path
from yazel_integration_sliding import recovar_pick_cleaner_sliding_batch, load_recovar_classifier

MODEL_PATH = '/mnt/data_a/ege/recovar_models/exp_instance/representation_learning_autoencoder_ensemble/instance/split0/ep19.h5'

PHASENET_THRESHOLD = 0.32

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

print("Processing with sliding windows in batches of 256...")
results = recovar_pick_cleaner_sliding_batch(streams=streams, classifier=classifier, batch_size=256)

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
        'scores_array': result['scores_array'],
        'mean_score': result['mean_score'],
        'max_score': result['max_score'],
        'detection_status': status
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n=== DETECTION STATISTICS ===\n")
print(f"Total windows: {len(comparison_df)}")

both = comparison_df[comparison_df['detection_status'] == 'Both']
phasenet_only = comparison_df[comparison_df['detection_status'] == 'PhaseNet only']

print(f"Both (catalog + PhaseNet): {len(both)}")
print(f"PhaseNet only: {len(phasenet_only)}")
print(f"Catalog only: {len(comparison_df[comparison_df['detection_status'] == 'Catalog only'])}\n")

print("=== MODEL SCORES (MEAN): BOTH (TRUE POSITIVES) ===")
both_mean_scores = both['mean_score'].values
print(f"Count: {len(both_mean_scores)}")
print(f"Mean: {np.mean(both_mean_scores):.3f}")
print(f"Std: {np.std(both_mean_scores):.3f}")
print(f"Min: {np.min(both_mean_scores):.3f}")
print(f"Max: {np.max(both_mean_scores):.3f}")

print("\n=== MODEL SCORES (MEAN): PHASENET ONLY (FALSE POSITIVES) ===")
phasenet_mean_scores = phasenet_only['mean_score'].values
print(f"Count: {len(phasenet_mean_scores)}")
print(f"Mean: {np.mean(phasenet_mean_scores):.3f}")
print(f"Std: {np.std(phasenet_mean_scores):.3f}")
print(f"Min: {np.min(phasenet_mean_scores):.3f}")
print(f"Max: {np.max(phasenet_mean_scores):.3f}")

print("\n=== MODEL SCORES (MAX): BOTH (TRUE POSITIVES) ===")
both_max_scores = both['max_score'].values
print(f"Count: {len(both_max_scores)}")
print(f"Mean: {np.mean(both_max_scores):.3f}")
print(f"Std: {np.std(both_max_scores):.3f}")
print(f"Min: {np.min(both_max_scores):.3f}")
print(f"Max: {np.max(both_max_scores):.3f}")

print("\n=== MODEL SCORES (MAX): PHASENET ONLY (FALSE POSITIVES) ===")
phasenet_max_scores = phasenet_only['max_score'].values
print(f"Count: {len(phasenet_max_scores)}")
print(f"Mean: {np.mean(phasenet_max_scores):.3f}")
print(f"Std: {np.std(phasenet_max_scores):.3f}")
print(f"Min: {np.min(phasenet_max_scores):.3f}")
print(f"Max: {np.max(phasenet_max_scores):.3f}")

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

print("\n=== FINDING BEST THRESHOLD (MEAN SCORES) ===")
all_mean_scores = comparison_df["mean_score"].values
all_labels = ~comparison_df["catalog_pick"].isna()

best_threshold_mean, best_f1_mean = find_best_f1_threshold(all_mean_scores, all_labels)
print(f"Best threshold: {best_threshold_mean:.6f}")
print(f"Best F1 score: {best_f1_mean:.3f}")

comparison_df['recovar_decision_mean'] = comparison_df['mean_score'] >= best_threshold_mean

tp_mean = np.sum((comparison_df['detection_status'] == 'Both') & comparison_df['recovar_decision_mean'])
fp_mean = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & comparison_df['recovar_decision_mean'])
fn_mean = np.sum((comparison_df['detection_status'] == 'Both') & ~comparison_df['recovar_decision_mean'])
tn_mean = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & ~comparison_df['recovar_decision_mean'])

precision_mean = tp_mean / (tp_mean + fp_mean) if (tp_mean + fp_mean) > 0 else 0
recall_mean = tp_mean / (tp_mean + fn_mean) if (tp_mean + fn_mean) > 0 else 0
f1_mean = 2 * precision_mean * recall_mean / (precision_mean + recall_mean) if (precision_mean + recall_mean) > 0 else 0

print(f"\n=== FINAL PERFORMANCE WITH MEAN THRESHOLD {best_threshold_mean:.6f} ===")
print(f"TP={tp_mean}, FP={fp_mean}, FN={fn_mean}, TN={tn_mean}")
print(f"Precision={precision_mean:.3f}, Recall={recall_mean:.3f}, F1={f1_mean:.3f}")

print("\n=== FINDING BEST THRESHOLD (MAX SCORES) ===")
all_max_scores = comparison_df["max_score"].values

best_threshold_max, best_f1_max = find_best_f1_threshold(all_max_scores, all_labels)
print(f"Best threshold: {best_threshold_max:.6f}")
print(f"Best F1 score: {best_f1_max:.3f}")

comparison_df['recovar_decision_max'] = comparison_df['max_score'] >= best_threshold_max

tp_max = np.sum((comparison_df['detection_status'] == 'Both') & comparison_df['recovar_decision_max'])
fp_max = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & comparison_df['recovar_decision_max'])
fn_max = np.sum((comparison_df['detection_status'] == 'Both') & ~comparison_df['recovar_decision_max'])
tn_max = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & ~comparison_df['recovar_decision_max'])

precision_max = tp_max / (tp_max + fp_max) if (tp_max + fp_max) > 0 else 0
recall_max = tp_max / (tp_max + fn_max) if (tp_max + fn_max) > 0 else 0
f1_max = 2 * precision_max * recall_max / (precision_max + recall_max) if (precision_max + recall_max) > 0 else 0

print(f"\n=== FINAL PERFORMANCE WITH MAX THRESHOLD {best_threshold_max:.6f} ===")
print(f"TP={tp_max}, FP={fp_max}, FN={fn_max}, TN={tn_max}")
print(f"Precision={precision_max:.3f}, Recall={recall_max:.3f}, F1={f1_max:.3f}")

manual_thresholds = [0.05,0.07,0.08]
print(f"\n=== PERFORMANCE AT MANUAL THRESHOLDS (MEAN SCORES) ===")
for manual_thr in manual_thresholds:
    comparison_df['manual_decision'] = comparison_df['mean_score'] >= manual_thr

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

print(f"\n=== PERFORMANCE AT MANUAL THRESHOLDS (MAX SCORES) ===")
for manual_thr in manual_thresholds:
    comparison_df['manual_decision'] = comparison_df['max_score'] >= manual_thr

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

comparison_df['scores_array_str'] = comparison_df['scores_array'].apply(lambda x: ','.join(map(str, x)))
comparison_df_to_save = comparison_df.drop(columns=['scores_array'])

comparison_df_to_save.to_csv('SLVT_pick_comparison_sliding.csv', index=False)
print(f"\nResults saved to: SLVT_pick_comparison_sliding.csv")

np.savez('SLVT_sliding_scores_arrays.npz',
         filenames=comparison_df['filename'].values,
         scores_arrays=np.array([arr for arr in comparison_df['scores_array'].values], dtype=object))
print(f"Score arrays saved to: SLVT_sliding_scores_arrays.npz")
