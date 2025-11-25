import obspy
import pandas as pd
import numpy as np
from pathlib import Path
from yazel_integration import recovar_pick_cleaner, recovar_pick_cleaner_batch, load_recovar_classifier

MODEL_PATH = '/mnt/data_a/ege/recovar_models/exp_instance/representation_learning_autoencoder_ensemble/instance/split0/ep19.h5'

phasenet_pick_dir = "/home/ege/recovar/reproducibility/phasenet_eqt/phasenet_windows_SLVT"
phasenet_picks = pd.read_csv("/home/ege/recovar/reproducibility/phasenet_eqt/phasenet_windows_SLVT/metadata.csv")

p = Path(phasenet_pick_dir)

print("Loading classifier...")
classifier = load_recovar_classifier(MODEL_PATH)

print("\n=== Testing OLD method (batch_size=1) ===")
old_results = []
files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix == '.mseed'])[:10]

for file in files:
    stream = obspy.read(file)
    stream.merge()
    result = recovar_pick_cleaner(stream=stream, classifier=classifier, pick_idx=None, threshold=None)
    old_results.append(result[0])
    print(f"{file.name}: {result[0]:.6f}")

print(f"\nOld method - Mean: {np.mean(old_results):.6f}, Std: {np.std(old_results):.6f}")

print("\n=== Testing NEW method (batch_size=256) ===")
streams = []
for file in files:
    stream = obspy.read(file)
    stream.merge()
    streams.append(stream)

batch_results = recovar_pick_cleaner_batch(streams=streams, classifier=classifier)

for i, (file, score) in enumerate(zip(files, batch_results)):
    print(f"{file.name}: {score:.6f}")

print(f"\nNew method - Mean: {np.mean(batch_results):.6f}, Std: {np.std(batch_results):.6f}")

print("\n=== Comparison ===")
print(f"Score difference: {np.mean(np.abs(np.array(old_results) - batch_results)):.6f}")
print("If scores are very different, batching matters significantly!")
