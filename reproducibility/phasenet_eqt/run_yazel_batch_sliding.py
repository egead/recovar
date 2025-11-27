import obspy
import pandas as pd
import numpy as np
from pathlib import Path
from yazel_integration_sliding import recovar_pick_cleaner_sliding_batch, load_recovar_classifier

MODEL_PATH = '/mnt/data_a/ege/recovar_models/exp_instance/representation_learning_autoencoder_ensemble/instance/split0/ep19.h5'

PHASENET_THRESHOLD = 0.50

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

analyzed_stations = comparison_df['station'].unique()

catalog_for_stations = catalog[catalog['station'].isin(analyzed_stations)]

total_catalog_picks = len(catalog_for_stations)

catalog_picks_detected = []
catalog_picks_missed = []

for idx, catalog_row in catalog_for_stations.iterrows():
    station = catalog_row['station']
    catalog_time = catalog_row['p_arrival_time']

    matching_windows = comparison_df[
        (comparison_df['station'] == station) &
        (comparison_df['catalog_pick'] == catalog_time)
    ]

    if len(matching_windows) > 0:
        catalog_picks_detected.append(catalog_row)
    else:
        catalog_picks_missed.append(catalog_row)

phasenet_detected = len(catalog_picks_detected)
phasenet_missed = len(catalog_picks_missed)
phasenet_detection_rate = (phasenet_detected / total_catalog_picks * 100) if total_catalog_picks > 0 else 0
phasenet_miss_rate = (phasenet_missed / total_catalog_picks * 100) if total_catalog_picks > 0 else 0

print("\n=== CATALOG STATISTICS ===\n")
print(f"Analyzed stations: {len(analyzed_stations)}")
print(f"Total catalog P-picks for these stations: {total_catalog_picks}")
print(f"PhaseNet detected (TP): {phasenet_detected} ({phasenet_detection_rate:.1f}%)")
print(f"PhaseNet missed (FN): {phasenet_missed} ({phasenet_miss_rate:.1f}%)")

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

manual_thresholds = [0.05,0.06,0.07,0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14,0.15,0.16,0.17,0.18,0.19,0.20]
print(f"\n=== PERFORMANCE AT MANUAL THRESHOLDS (MEAN SCORES) ===")
mean_manual_results = []
for manual_thr in manual_thresholds:
    comparison_df['manual_decision'] = comparison_df['mean_score'] >= manual_thr

    tp_m = np.sum((comparison_df['detection_status'] == 'Both') & comparison_df['manual_decision'])
    fp_m = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & comparison_df['manual_decision'])
    fn_m = np.sum((comparison_df['detection_status'] == 'Both') & ~comparison_df['manual_decision'])
    tn_m = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & ~comparison_df['manual_decision'])

    precision_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0
    recall_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
    f1_m = 2 * precision_m * recall_m / (precision_m + recall_m) if (precision_m + recall_m) > 0 else 0

    mean_manual_results.append({
        'threshold': manual_thr,
        'tp': tp_m,
        'fp': fp_m,
        'fn': fn_m,
        'tn': tn_m,
        'precision': precision_m,
        'recall': recall_m,
        'f1': f1_m
    })

    print(f"\nThreshold: {manual_thr:.2f}")
    print(f"  TP={tp_m}, FP={fp_m}, FN={fn_m}, TN={tn_m}")
    print(f"  Precision={precision_m:.3f}, Recall={recall_m:.3f}, F1={f1_m:.3f}")

print(f"\n=== PERFORMANCE AT MANUAL THRESHOLDS (MAX SCORES) ===")
max_manual_results = []
for manual_thr in manual_thresholds:
    comparison_df['manual_decision'] = comparison_df['max_score'] >= manual_thr

    tp_m = np.sum((comparison_df['detection_status'] == 'Both') & comparison_df['manual_decision'])
    fp_m = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & comparison_df['manual_decision'])
    fn_m = np.sum((comparison_df['detection_status'] == 'Both') & ~comparison_df['manual_decision'])
    tn_m = np.sum((comparison_df['detection_status'] == 'PhaseNet only') & ~comparison_df['manual_decision'])

    precision_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0
    recall_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
    f1_m = 2 * precision_m * recall_m / (precision_m + recall_m) if (precision_m + recall_m) > 0 else 0

    max_manual_results.append({
        'threshold': manual_thr,
        'tp': tp_m,
        'fp': fp_m,
        'fn': fn_m,
        'tn': tn_m,
        'precision': precision_m,
        'recall': recall_m,
        'f1': f1_m
    })

    print(f"\nThreshold: {manual_thr:.2f}")
    print(f"  TP={tp_m}, FP={fp_m}, FN={fn_m}, TN={tn_m}")
    print(f"  Precision={precision_m:.3f}, Recall={recall_m:.3f}, F1={f1_m:.3f}")

def generate_report(phasenet_threshold, comparison_df, both, phasenet_only,
                   best_threshold_mean, tp_mean, fp_mean, fn_mean, tn_mean,
                   precision_mean, recall_mean, f1_mean, mean_manual_results,
                   best_threshold_max, tp_max, fp_max, fn_max, tn_max,
                   precision_max, recall_max, f1_max, max_manual_results,
                   total_catalog_picks, phasenet_detected, phasenet_missed,
                   phasenet_detection_rate, phasenet_miss_rate):
    """Generate a formatted report and save to text file."""

    report_lines = []
    report_lines.append("="*70)
    report_lines.append("YAZEL BATCH SLIDING WINDOW ANALYSIS REPORT")
    report_lines.append("="*70)
    report_lines.append("")

    report_lines.append("LEGEND:")
    report_lines.append("-"*70)
    report_lines.append("CATALOG STATISTICS:")
    report_lines.append("  - Shows PhaseNet's performance against ground truth catalog")
    report_lines.append("  - PhaseNet detected: Catalog events that PhaseNet found")
    report_lines.append("  - PhaseNet missed: Catalog events that PhaseNet completely missed")
    report_lines.append("  - These events cannot be recovered by RECOVAR filtering")
    report_lines.append("")
    report_lines.append("RECOVAR FILTER PERFORMANCE (TP/FP/FN/TN):")
    report_lines.append("  - Evaluates RECOVAR's ability to filter PhaseNet picks")
    report_lines.append("  - TP (True Positive): PhaseNet pick with catalog event, RECOVAR kept it")
    report_lines.append("  - FP (False Positive): PhaseNet pick without catalog event, RECOVAR kept it")
    report_lines.append("  - FN (False Negative): PhaseNet pick with catalog event, RECOVAR rejected it")
    report_lines.append("  - TN (True Negative): PhaseNet pick without catalog event, RECOVAR rejected it")
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("")

    report_lines.append(f"PhaseNet Threshold: {phasenet_threshold}")
    report_lines.append("Sliding window: 60 seconds with 1 sec iterations")
    report_lines.append("")

    report_lines.append("-"*70)
    report_lines.append("CATALOG STATISTICS (PhaseNet Baseline Performance)")
    report_lines.append("-"*70)
    report_lines.append(f"    Total catalog P-picks for analyzed stations: {total_catalog_picks}")
    report_lines.append(f"    PhaseNet detected (True Positives): {phasenet_detected} ({phasenet_detection_rate:.1f}%)")
    report_lines.append(f"    PhaseNet missed (False Negatives): {phasenet_missed} ({phasenet_miss_rate:.1f}%)")
    report_lines.append("")

    report_lines.append("-"*70)
    report_lines.append("DETECTION STATISTICS")
    report_lines.append("-"*70)
    report_lines.append(f"    Total windows: {len(comparison_df)}")
    report_lines.append(f"    Both (catalog + PhaseNet): {len(both)}")
    report_lines.append(f"    PhaseNet only: {len(phasenet_only)}")
    report_lines.append(f"    Catalog only: {len(comparison_df[comparison_df['detection_status'] == 'Catalog only'])}")
    report_lines.append("")

    report_lines.append("="*70)
    report_lines.append("SLIDING WINDOW MEAN SCORES")
    report_lines.append("="*70)

    total_fps = len(phasenet_only)
    total_tps = len(both)
    fps_filtered_mean = (tn_mean / total_fps * 100) if total_fps > 0 else 0
    tps_lost_mean = (fn_mean / total_tps * 100) if total_tps > 0 else 0

    report_lines.append(f"    Best threshold: {best_threshold_mean:.3f}")
    report_lines.append(f"    Filter out {fps_filtered_mean:.1f}% of FPs, lose {tps_lost_mean:.1f}% of TPs")
    report_lines.append(f"    TP={tp_mean}, FP={fp_mean}, FN={fn_mean}, TN={tn_mean}")
    report_lines.append(f"    Precision={precision_mean:.3f}, Recall={recall_mean:.3f}, F1={f1_mean:.3f} (best F1 score)")
    report_lines.append("")
    report_lines.append("    Manual thresholds:")

    for result in mean_manual_results:
        fps_filtered = (result['tn'] / total_fps * 100) if total_fps > 0 else 0
        tps_lost = (result['fn'] / total_tps * 100) if total_tps > 0 else 0
        report_lines.append(f"    Threshold {result['threshold']:.2f}: Filter out {fps_filtered:.1f}% of FPs, lose {tps_lost:>5.1f}% of TPs  (F1={result['f1']:.3f})")

    report_lines.append("")

    report_lines.append("="*70)
    report_lines.append("SLIDING WINDOW MAX SCORES")
    report_lines.append("="*70)

    fps_filtered_max = (tn_max / total_fps * 100) if total_fps > 0 else 0
    tps_lost_max = (fn_max / total_tps * 100) if total_tps > 0 else 0

    report_lines.append(f"    Best threshold: {best_threshold_max:.3f}")
    report_lines.append(f"    Filter out {fps_filtered_max:.1f}% of FPs, lose {tps_lost_max:.1f}% of TPs")
    report_lines.append(f"    TP={tp_max}, FP={fp_max}, FN={fn_max}, TN={tn_max}")
    report_lines.append(f"    Precision={precision_max:.3f}, Recall={recall_max:.3f}, F1={f1_max:.3f} (best F1 score)")
    report_lines.append("")
    report_lines.append("    Manual thresholds:")

    for result in max_manual_results:
        fps_filtered = (result['tn'] / total_fps * 100) if total_fps > 0 else 0
        tps_lost = (result['fn'] / total_tps * 100) if total_tps > 0 else 0
        report_lines.append(f"    Threshold {result['threshold']:.2f}: Filter out {fps_filtered:.1f}% of FPs, lose {tps_lost:>5.1f}% of TPs  (F1={result['f1']:.3f})")

    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("END OF REPORT")
    report_lines.append("="*70)

    return "\n".join(report_lines)

report_text = generate_report(
    PHASENET_THRESHOLD, comparison_df, both, phasenet_only,
    best_threshold_mean, tp_mean, fp_mean, fn_mean, tn_mean,
    precision_mean, recall_mean, f1_mean, mean_manual_results,
    best_threshold_max, tp_max, fp_max, fn_max, tn_max,
    precision_max, recall_max, f1_max, max_manual_results,
    total_catalog_picks, phasenet_detected, phasenet_missed,
    phasenet_detection_rate, phasenet_miss_rate
)

report_filename = f'SLVT_yazel_report_thr_{PHASENET_THRESHOLD:.2f}.txt'
with open(report_filename, 'w') as f:
    f.write(report_text)

print("\n" + "="*70)
print(report_text)
print(f"\n\nReport saved to: {report_filename}")

comparison_df['scores_array_str'] = comparison_df['scores_array'].apply(lambda x: ','.join(map(str, x)))
comparison_df_to_save = comparison_df.drop(columns=['scores_array'])

comparison_df_to_save.to_csv('SLVT_pick_comparison_sliding.csv', index=False)
print(f"Results saved to: SLVT_pick_comparison_sliding.csv")

np.savez('SLVT_sliding_scores_arrays.npz',
         filenames=comparison_df['filename'].values,
         scores_arrays=np.array([arr for arr in comparison_df['scores_array'].values], dtype=object))
print(f"Score arrays saved to: SLVT_sliding_scores_arrays.npz")
