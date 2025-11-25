
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from latent_space_visualization_new import plot_latent_samples
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from zoneinfo import ZoneInfo
from recovar.representation_learning_models import RepresentationLearningMultipleAutoencoder
from recovar.classifier_models import ClassifierMultipleAutoencoder

########################################################
NPY_DIR = '/home/ege/10NOV_2-3/'
instance='/mnt/data_a/ege/recovar_models/exp_instance/representation_learning_autoencoder_ensemble/instance/split0/ep19.h5'
merged='/mnt/data_a/ege/recovar_models/MERGED_dilation_v2/representation_learning_autoencoder_ensemble/MERGED_fixed/split0/ep9.h5'
MODEL_PATH =instance
OUTPUT_DIR = '10NOV_2-3_INSTANCE_1SEC'
STRIDE =1
WINDOW_SIZE = 3000
SAMPLING_RATE = 100
START_TIME = datetime(2025, 11, 10, 1, 50)
BATCH_SIZE = 1024
TIMEZONE = 'Europe/Istanbul'
localized_start_time = START_TIME.replace(tzinfo=ZoneInfo(TIMEZONE))

SAVE_LATENT_PLOTS = True
LATENT_PLOT_INTERVAL =30  #Save latent plot every N seconds
LATENT_SAMPLES_PER_BATCH = 3  #Number of samples to plot from each saved batch

#Time range for extracting latent samples (None = all samples)
LATENT_START_TIME = datetime(2025, 11, 10, 1, 50)
LATENT_END_TIME = datetime(2025, 11, 10, 2, 00)

PLOT_START_TIME = datetime(2025, 11, 10, 1, 50)
PLOT_END_TIME = datetime(2025, 11, 10, 3, 00)

CATALOG_DATA =  """10/11/2025 03:09:26,2.9,Sındırgı (Balıkesir)
10/11/2025 02:57:20,2.8,Sındırgı (Balıkesir)
10/11/2025 02:52:53,3.7,Sındırgı (Balıkesir)
10/11/2025 02:48:57,4.8,Sındırgı (Balıkesir)
10/11/2025 02:26:54,2.6,Sındırgı (Balıkesir)
10/11/2025 02:07:29,2.4,Sındırgı (Balıkesir)
10/11/2025 01:56:47,2.7,Sındırgı (Balıkesir)"""
########################################################
print("Loading model...")
model = RepresentationLearningMultipleAutoencoder()
model.compile()
model(np.random.randn(1, WINDOW_SIZE, 3).astype(np.float32))
model.load_weights(MODEL_PATH)
classifier = ClassifierMultipleAutoencoder(model)
print("Model loaded")

print("\nParsing earthquake catalog...")
from io import StringIO
catalog_df = pd.read_csv(StringIO(CATALOG_DATA.strip()), header=None, names=['datetime', 'magnitude', 'location'])
catalog_df['datetime'] = pd.to_datetime(catalog_df['datetime'], format='%d/%m/%Y %H:%M:%S')
catalog_df['datetime'] = catalog_df['datetime'].dt.tz_localize(TIMEZONE if TIMEZONE else 'UTC')
earthquake_times = catalog_df['datetime'].tolist()
print(f"Loaded {len(earthquake_times)} earthquake times from catalog")
print(f"First few earthquake times: {earthquake_times[:3]}")
print(f"Last few earthquake times: {earthquake_times[-3:]}")

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

npy_dir = Path(NPY_DIR)
npy_files = list(npy_dir.glob("*.npy"))
print(f"\nFound {len(npy_files)} .npy files")

if len(npy_files) == 0:
    print("No .npy files found!")
    exit(1)

stride_samples = int(STRIDE * SAMPLING_RATE)

all_results = {}
all_waveforms = {}
########################################################
for npy_file in npy_files:
    station_name = npy_file.stem
    print(f"\n{'='*60}")
    print(f"Processing: {station_name}")
    print(f"{'='*60}")

    print(f"Loading: {npy_file}")
    waveform = np.load(npy_file)
    print(f"Shape: {waveform.shape}")

    all_waveforms[station_name] = waveform


    start_time = START_TIME
    if TIMEZONE and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=ZoneInfo(TIMEZONE))

    results = []
    batch = []
    batch_meta = []

    # Track when to save latent plots
    last_latent_save_time = None

    print("Running inference...")
    for i in range(0, waveform.shape[0] - WINDOW_SIZE + 1, stride_samples):
        window = waveform[i:i+WINDOW_SIZE, :]

        if np.isnan(window).any() or np.isinf(window).any():
            continue

        window = window / (np.max(np.abs(window)) + 1e-6)

        batch.append(window)

        #window_time = start_time + timedelta(seconds=i/SAMPLING_RATE)
        #ASSIGNS THE SCORE TO WINDOW CENTER, NOT WINDOW BEGINNING TO REDUCE THE LATENESS BIAS
        #BECAUSE CURRENTLY, THE CODE ASSIGNS SCORE AT i=0, IF EVENT HAPPENS LATER IN THE WINDOW
        #WE HAVE A HUGE DISADVANTAGE AGAINST EQT AND PHASENET, BECAUSE THEY SCORE BY SAMPLE
        #NOT BY WINDOW
        window_time = start_time + timedelta(seconds=(i + WINDOW_SIZE/2)/SAMPLING_RATE)
        batch_meta.append({'timestamp': window_time, 'sample': i})

        if len(batch) == BATCH_SIZE:
            scores = classifier(np.array(batch, dtype=np.float32)).flatten()

            # Check if we should save latent plots for this batch
            current_time = batch_meta[0]['timestamp']
            should_save_plots = False

            if SAVE_LATENT_PLOTS:
                # Check if current time is within the specified latent extraction range
                in_time_range = True
                if LATENT_START_TIME is not None or LATENT_END_TIME is not None:
                    latent_start = LATENT_START_TIME.replace(tzinfo=ZoneInfo(TIMEZONE)) if LATENT_START_TIME and LATENT_START_TIME.tzinfo is None else LATENT_START_TIME
                    latent_end = LATENT_END_TIME.replace(tzinfo=ZoneInfo(TIMEZONE)) if LATENT_END_TIME and LATENT_END_TIME.tzinfo is None else LATENT_END_TIME

                    if latent_start is not None and current_time < latent_start:
                        in_time_range = False
                    if latent_end is not None and current_time > latent_end:
                        in_time_range = False

                if in_time_range:
                    if last_latent_save_time is None:
                        should_save_plots = True
                    else:
                        time_since_last_save = (current_time - last_latent_save_time).total_seconds()
                        if time_since_last_save >= LATENT_PLOT_INTERVAL:
                            should_save_plots = True
            
            #GET FEATURE MAPS
            if should_save_plots:
                f1p, f2p, f3p, f4p, f5p, __, __, __, __, __ = model(np.array(batch, dtype=np.float32))

                latent_dir = output_dir / 'latent_plots' / station_name
                latent_dir.mkdir(parents=True, exist_ok=True)

                # Plot multiple samples from the batch
                num_samples = min(LATENT_SAMPLES_PER_BATCH, len(batch))
                indices = np.linspace(0, len(batch) - 1, num_samples, dtype=int)

                for idx in indices:
                    timestamp_str = batch_meta[idx]['timestamp'].strftime('%Y%m%d_%H%M%S')
                    sample_idx = batch_meta[idx]['sample']
                    plot_path = latent_dir / f"latent_s{sample_idx}_{timestamp_str}.png"

                    plot_latent_samples(waveform=batch[idx],
                                       feature_maps=[f1p[idx], f2p[idx], f3p[idx], f4p[idx], f5p[idx]],
                                       path=plot_path)

                last_latent_save_time = current_time
                print(f"\nSaved {num_samples} latent plots at {current_time.strftime('%H:%M:%S')}")

            for meta, score in zip(batch_meta, scores):
                results.append({**meta, 'eq_probability': score})
            print(f"Processed {len(results)} windows", end='\r')
            batch = []
            batch_meta = []

    if batch:
        scores = classifier(np.array(batch, dtype=np.float32)).flatten()
        f1p, f2p, f3p, f4p, f5p, __, __, __, __, __ = model(np.array(batch, dtype=np.float32))

        for meta, score in zip(batch_meta, scores):
            results.append({**meta, 'eq_probability': score})

        # Check if we should save latent plots for this batch
        current_time = batch_meta[0]['timestamp']
        should_save_plots = False

        if SAVE_LATENT_PLOTS:
            # Check if current time is within the specified latent extraction range
            in_time_range = True
            if LATENT_START_TIME is not None or LATENT_END_TIME is not None:
                latent_start = LATENT_START_TIME.replace(tzinfo=ZoneInfo(TIMEZONE)) if LATENT_START_TIME and LATENT_START_TIME.tzinfo is None else LATENT_START_TIME
                latent_end = LATENT_END_TIME.replace(tzinfo=ZoneInfo(TIMEZONE)) if LATENT_END_TIME and LATENT_END_TIME.tzinfo is None else LATENT_END_TIME

                if latent_start is not None and current_time < latent_start:
                    in_time_range = False
                if latent_end is not None and current_time > latent_end:
                    in_time_range = False

            if in_time_range:
                if last_latent_save_time is None:
                    should_save_plots = True
                else:
                    time_since_last_save = (current_time - last_latent_save_time).total_seconds()
                    if time_since_last_save >= LATENT_PLOT_INTERVAL:
                        should_save_plots = True

        if should_save_plots:
            # Create subdirectory for latent plots
            latent_dir = output_dir / 'latent_plots' / station_name
            latent_dir.mkdir(parents=True, exist_ok=True)

            # Plot multiple samples from the batch
            num_samples = min(LATENT_SAMPLES_PER_BATCH, len(batch))
            indices = np.linspace(0, len(batch) - 1, num_samples, dtype=int)

            for idx in indices:
                timestamp_str = batch_meta[idx]['timestamp'].strftime('%Y%m%d_%H%M%S')
                sample_idx = batch_meta[idx]['sample']
                plot_path = latent_dir / f"latent_s{sample_idx}_{timestamp_str}.png"

                plot_latent_samples(waveform=batch[idx],
                                   feature_maps=[f1p[idx], f2p[idx], f3p[idx], f4p[idx], f5p[idx]],
                                   path=plot_path)

            last_latent_save_time = current_time
            print(f"\nSaved {num_samples} latent plots at {current_time.strftime('%H:%M:%S')}")

    print(f"\nDone: {len(results)} predictions")

    df = pd.DataFrame(results)
    csv_path = output_dir / f"{station_name}_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    series = pd.Series(data=df['eq_probability'].values, index=df['timestamp'], name=station_name)
    all_results[station_name] = series

    print(f"Mean: {df['eq_probability'].mean():.4f}, Max: {df['eq_probability'].max():.4f}")
########################################################
print(f"\n{'='*60}")
print("Creating plots...")
print(f"{'='*60}")

if TIMEZONE:
    tz = ZoneInfo(TIMEZONE)
    time_formatter = mdates.DateFormatter('%H:%M', tz=tz)
    PLOT_START_TIME = PLOT_START_TIME.replace(tzinfo=tz)
    PLOT_END_TIME = PLOT_END_TIME.replace(tzinfo=tz)
else:
    time_formatter = mdates.DateFormatter('%H:%M')

plt.figure(figsize=(14, 7))
ax = plt.gca()
ax.xaxis.set_major_formatter(time_formatter)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

for station_name in sorted(all_results.keys()):
    series = all_results[station_name]
    ax.fill_between(series.index, 0, series.values, alpha=0.3, label=f'{station_name} (prob)')
    ax.plot(series.index, series.values, alpha=0.7, linewidth=1)

ax2 = ax.twinx()
for station_name in sorted(all_waveforms.keys()):
    waveform = all_waveforms[station_name]
    n_samples = waveform.shape[0]
    sample_times = [localized_start_time + timedelta(seconds=i/SAMPLING_RATE) for i in range(n_samples)]
    #Plot Z channel as representative waveform
    z_channel = waveform[:, 2]
    ax2.plot(sample_times, z_channel, alpha=0.5, linewidth=0.3, color='gray')

ax2.set_ylabel('Waveform Amplitude', fontsize=9, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

for idx, eq_time in enumerate(earthquake_times):
    label = 'Catalog EQ' if idx == 0 else None
    ax.axvline(eq_time, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label=label)

plt.xlim(PLOT_START_TIME, PLOT_END_TIME)
ax.set_ylabel('Event Score')
ax.set_xlabel('Time')
ax.set_ylim(0, 1)
plt.title('Event Scores with Waveforms - All Stations')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()

full_plot_path = output_dir / 'all_stations_full.png'
plt.savefig(full_plot_path, dpi=150, bbox_inches='tight')
print(f"Saved: {full_plot_path}")
plt.show()
########################################################
for station_name in sorted(all_results.keys()):
    if station_name not in all_waveforms:
        continue

    plt.figure(figsize=(14, 5))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(time_formatter)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    series = all_results[station_name]
    ax.fill_between(series.index, 0, series.values, alpha=0.3, color='steelblue', label=f'{station_name} (prob)')
    ax.plot(series.index, series.values, alpha=0.8, linewidth=1.5, color='steelblue')

    ax2 = ax.twinx()
    waveform = all_waveforms[station_name]
    n_samples = waveform.shape[0]
    sample_times = [localized_start_time + timedelta(seconds=i/SAMPLING_RATE) for i in range(n_samples)]
    #Plot Z channel as representative waveform
    z_channel = waveform[:, 2]
    ax2.plot(sample_times, z_channel, alpha=0.5, linewidth=0.3, color='gray', label='Waveform (Z)')

    ax2.set_ylabel('Waveform Amplitude', fontsize=9, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    for idx, eq_time in enumerate(earthquake_times):
        label = 'Catalog EQ' if idx == 0 else None
        ax.axvline(eq_time, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label=label)

    plt.xlim(PLOT_START_TIME, PLOT_END_TIME)
    ax.set_ylabel('Event Score')
    ax.set_xlabel('Time')
    ax.set_ylim(0, 1)
    plt.title(f'Event Scores with Waveform - {station_name}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    station_plot_path = output_dir / f'{station_name}_plot.png'
    plt.savefig(station_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {station_plot_path}")
    plt.show()

print("\nCreating waveform plots with probability shadows...")
########################################################
def create_waveform_with_probability_plot(station_name, waveform, predictions, start_time,
                                          time_range=None, output_path=None, title_suffix="",
                                          earthquake_times=None):

    n_samples = waveform.shape[0]
    sample_times = [start_time + timedelta(seconds=i/SAMPLING_RATE) for i in range(n_samples)]

    if time_range:
        if time_range[0].tzinfo is not None and (len(sample_times) > 0 and sample_times[0].tzinfo is None):
            sample_times = [t.replace(tzinfo=time_range[0].tzinfo) for t in sample_times]

        start_idx = 0
        end_idx = n_samples
        for i, t in enumerate(sample_times):
            if t >= time_range[0]:
                start_idx = i
                break
        for i in range(len(sample_times)-1, -1, -1):
            if sample_times[i] <= time_range[1]:
                end_idx = i + 1
                break

        waveform = waveform[start_idx:end_idx]
        sample_times = sample_times[start_idx:end_idx]

        predictions = predictions[(predictions.index >= time_range[0]) & (predictions.index <= time_range[1])]

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    channel_names = ['East (E)', 'North (N)', 'Vertical (Z)']
    colors = ['#d62728', '#2ca02c', '#1f77b4']

    for i, (ax, channel_name, color) in enumerate(zip(axes[:3], channel_names, colors)):
        ax.plot(sample_times, waveform[:, i], color=color, linewidth=0.5, alpha=0.8)

        ax2 = ax.twinx()
        ax2.fill_between(predictions.index, 0, predictions.values,
                         color='orange', alpha=0.2, label='EQ Probability')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Probability', fontsize=9, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange', labelsize=8)

        ax.set_ylabel(f'{channel_name}\nAmplitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelsize=8)

        if earthquake_times is not None:
            for idx, eq_time in enumerate(earthquake_times):
                if time_range is None or (eq_time >= time_range[0] and eq_time <= time_range[1]):
                    label = 'Catalog EQ' if idx == 0 and i == 0 else None
                    ax.axvline(eq_time, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label=label)

        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    ax_prob = axes[3]
    ax_prob.fill_between(predictions.index, 0, predictions.values,
                         color='orange', alpha=0.4)
    ax_prob.plot(predictions.index, predictions.values, color='darkorange', linewidth=1)

    if earthquake_times is not None:
        for eq_time in earthquake_times:
            if time_range is None or (eq_time >= time_range[0] and eq_time <= time_range[1]):
                ax_prob.axvline(eq_time, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)

    ax_prob.set_ylabel('EQ Probability', fontsize=9)
    ax_prob.set_xlabel('Time', fontsize=10)
    ax_prob.set_ylim(0, 1)
    ax_prob.grid(True, alpha=0.3)
    ax_prob.tick_params(labelsize=8)

    ax_prob.xaxis.set_major_formatter(time_formatter)
    if time_range:
        duration = (time_range[1] - time_range[0]).total_seconds() / 3600
        if duration < 2:
            ax_prob.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        elif duration < 4:
            ax_prob.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        else:
            ax_prob.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    else:
        ax_prob.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    plt.setp(ax_prob.xaxis.get_majorticklabels(), rotation=45, ha='right')

    title = f'Waveforms and EQ Probability - {station_name}'
    if title_suffix:
        title += f' {title_suffix}'

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.show()

########################################################

for station_name in sorted(all_results.keys()):
    if station_name not in all_waveforms:
        continue

    waveform = all_waveforms[station_name]
    predictions = all_results[station_name]

    print(f"\nPlotting waveforms for {station_name} (full range)...")
    create_waveform_with_probability_plot(
        station_name=station_name,
        waveform=waveform,
        predictions=predictions,
        start_time=localized_start_time,
        time_range=(PLOT_START_TIME, PLOT_END_TIME),
        output_path=output_dir / f'{station_name}_waveform_full.png',
        title_suffix='(Full Range)',
        earthquake_times=earthquake_times
    )

print(f"\n{'='*60}")
print("All done!")
print(f"{'='*60}")