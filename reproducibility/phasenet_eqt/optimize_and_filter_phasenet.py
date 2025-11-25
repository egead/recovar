from phasenet_picker import extract_windows_phasenet
from collections import defaultdict
import seisbench.models as sbm
import pandas as pd
import numpy as np
import obspy
from pathlib import Path
import os
import shutil
from sklearn.metrics import f1_score

LOW_THRESHOLD = 0.3

print("Loading PhaseNet model...")
model = sbm.PhaseNet.from_pretrained("instance")

catalog_path = '/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/processed_catalogs/kara74a_phase_picks.csv'
catalog = pd.read_csv(catalog_path)
catalog['p_arrival_time'] = pd.to_datetime(catalog['p_arrival_time'])

station_dir = Path('/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/prepared_waveforms/day_by_day/SLVT')
files = sorted(station_dir.glob('*.mseed'))

date_files = defaultdict(list)
for f in files:
    date = f.name.split('__')[1][:8]
    date_files[date].append(f)

temp_phasenet_picks_dir = 'temp_phasenet_picks_dir'
os.makedirs(temp_phasenet_picks_dir, exist_ok=True)

all_metadata = []

print(f"Extracting windows with low threshold={LOW_THRESHOLD}...")
for date in sorted(date_files.keys()):
    print(f"Processing {date}")
    stream = obspy.Stream()
    for f in date_files[date]:
        stream += obspy.read(str(f))
    stream.merge(method=1, fill_value=0)

    windows, pick_times, peak_probs = extract_windows_phasenet(
        stream,
        model=model,
        phase='P',
        threshold=LOW_THRESHOLD
    )

    for i, (window, pick_time, prob) in enumerate(zip(windows, pick_times, peak_probs)):
        filename = f'window_{date}_{i:04d}.mseed'
        window.write(os.path.join(temp_phasenet_picks_dir, filename), format='MSEED')

        all_metadata.append({
            'phase': 'P',
            'date': date,
            'filename': filename,
            'pick_time': pick_time.isoformat(),
            'phasenet_prob': prob,
            'station': window[0].stats.station,
            'network': window[0].stats.network,
            'start_time': window[0].stats.starttime.isoformat(),
            'end_time': window[0].stats.endtime.isoformat()
        })

df = pd.DataFrame(all_metadata)

print(f"\nLabeling picks against catalog...")
df['pick_time'] = pd.to_datetime(df['pick_time'], format='ISO8601')
df['start_time'] = pd.to_datetime(df['start_time'], format='ISO8601')
df['end_time'] = pd.to_datetime(df['end_time'], format='ISO8601')

labels = []
for _, pick_row in df.iterrows():
    station = pick_row['station']
    window_start = pick_row['start_time']
    window_end = pick_row['end_time']

    catalog_picks = catalog[
        (catalog['station'] == station) &
        (catalog['p_arrival_time'] >= window_start) &
        (catalog['p_arrival_time'] <= window_end)
    ]

    has_catalog = not catalog_picks.empty
    labels.append(1 if has_catalog else 0)

df['has_catalog'] = labels

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

scores = df['phasenet_prob'].values
labels = df['has_catalog'].values

print(f"\n=== FINDING OPTIMAL PHASENET THRESHOLD ===")
best_threshold, best_f1 = find_best_f1_threshold(scores, labels)

tp_scores = scores[labels == 1]
fp_scores = scores[labels == 0]

print(f"\nTotal extracted picks: {len(df)}")
print(f"True Positives: {np.sum(labels == 1)}, Mean={np.mean(tp_scores):.3f}, Std={np.std(tp_scores):.3f}")
print(f"False Positives: {np.sum(labels == 0)}, Mean={np.mean(fp_scores):.3f}, Std={np.std(fp_scores):.3f}")
print(f"\nBest threshold: {best_threshold:.4f}")
print(f"Best F1 score: {best_f1:.3f}")

tp = np.sum((labels == 1) & (scores >= best_threshold))
fp = np.sum((labels == 0) & (scores >= best_threshold))
fn = np.sum((labels == 1) & (scores < best_threshold))
tn = np.sum((labels == 0) & (scores < best_threshold))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n=== FINAL PERFORMANCE WITH THRESHOLD {best_threshold:.4f} ===")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={best_f1:.3f}")

manual_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\n=== PERFORMANCE AT MANUAL THRESHOLDS ===")
for manual_thr in manual_thresholds:
    tp_m = np.sum((labels == 1) & (scores >= manual_thr))
    fp_m = np.sum((labels == 0) & (scores >= manual_thr))
    fn_m = np.sum((labels == 1) & (scores < manual_thr))
    tn_m = np.sum((labels == 0) & (scores < manual_thr))

    precision_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0
    recall_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
    f1_m = 2 * precision_m * recall_m / (precision_m + recall_m) if (precision_m + recall_m) > 0 else 0

    print(f"\nThreshold: {manual_thr:.2f}")
    print(f"  TP={tp_m}, FP={fp_m}, FN={fn_m}, TN={tn_m}")
    print(f"  Precision={precision_m:.3f}, Recall={recall_m:.3f}, F1={f1_m:.3f}")

print(f"\n=== FILTERING AND COPYING FILES ===")
filtered_df = df[df['phasenet_prob'] >= best_threshold].copy()

filtered_phasenet_picks_dir = 'filtered_phasenet_picks_dir'
os.makedirs(filtered_phasenet_picks_dir, exist_ok=True)

print(f"Copying {len(filtered_df)} filtered windows (optimal threshold {best_threshold:.4f})...")
for filename in filtered_df['filename']:
    src = os.path.join(temp_phasenet_picks_dir, filename)
    dst = os.path.join(filtered_phasenet_picks_dir, filename)
    shutil.copy(src, dst)

filtered_df.to_csv(os.path.join(filtered_phasenet_picks_dir, 'metadata.csv'), index=False)

for manual_thr in manual_thresholds:
    manual_filtered_df = df[df['phasenet_prob'] >= manual_thr].copy()
    manual_dir = f'filtered_phasenet_picks_dir_thr_{manual_thr:.2f}'
    os.makedirs(manual_dir, exist_ok=True)

    print(f"Copying {len(manual_filtered_df)} filtered windows (threshold {manual_thr:.2f})...")
    for filename in manual_filtered_df['filename']:
        src = os.path.join(temp_phasenet_picks_dir, filename)
        dst = os.path.join(manual_dir, filename)
        shutil.copy(src, dst)

    manual_filtered_df.to_csv(os.path.join(manual_dir, 'metadata.csv'), index=False)

print(f"\nCleaning up temporary directory...")
shutil.rmtree(temp_phasenet_picks_dir)

print(f"\n=== DONE ===")
print(f"Filtered {len(df)} picks → {len(filtered_df)} picks (optimal threshold)")
print(f"Reduction: {100 * (1 - len(filtered_df)/len(df)):.1f}%")
print(f"Saved to: {filtered_phasenet_picks_dir}/")
print(f"Optimal PhaseNet threshold: {best_threshold:.4f}")
print(f"\nAdditional filtered datasets saved for manual thresholds:")
for manual_thr in manual_thresholds:
    manual_count = len(df[df['phasenet_prob'] >= manual_thr])
    print(f"  Threshold {manual_thr:.2f}: {manual_count} picks → filtered_phasenet_picks_dir_thr_{manual_thr:.2f}/")
