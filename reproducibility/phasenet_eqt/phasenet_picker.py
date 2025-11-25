import numpy as np
import obspy
import pandas as pd
import os
import seisbench.models as sbm
from scipy.signal import find_peaks

def extract_windows_phasenet(stream, model_name='instance', threshold=0.7, phase='P', window_samples=4500):
    model = sbm.PhaseNet.from_pretrained(model_name)

    stream_sync = stream.copy()
    stream_sync.merge(method=1, fill_value=0)
    stream_sync.trim(
        starttime=max([tr.stats.starttime for tr in stream_sync]),
        endtime=min([tr.stats.endtime for tr in stream_sync]),
        pad=True,
        fill_value=0
    )

    annotations = model.annotate(stream_sync)
    phase_channel = [tr for tr in annotations if tr.stats.channel.endswith(phase)][0]
    peaks, _ = find_peaks(phase_channel.data, height=threshold, distance=100)

    windows = []
    pick_times = []
    half_samples = window_samples // 2

    for peak_idx in peaks:
        pick_time = phase_channel.stats.starttime + peak_idx / phase_channel.stats.sampling_rate
        offset_seconds = pick_time - stream_sync[0].stats.starttime
        stream_pick_sample = int(round(offset_seconds * stream_sync[0].stats.sampling_rate))

        start_sample = stream_pick_sample - half_samples
        end_sample = start_sample + window_samples

        if start_sample >= 0 and end_sample <= len(stream_sync[0].data):
            windowed_stream = obspy.Stream()
            for tr in stream_sync:
                windowed = tr.copy()
                windowed.data = tr.data[start_sample:end_sample]
                windowed.stats.starttime = tr.stats.starttime + start_sample / tr.stats.sampling_rate
                windowed_stream.append(windowed)

            windows.append(windowed_stream)
            pick_times.append(pick_time)

    return windows, pick_times

def save_windows(windows, pick_times, output_dir='phasenet_windows', metadata_filename='metadata.csv'):
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    for i, (window, pick_time) in enumerate(zip(windows, pick_times)):
        filename = f'window_{i:04d}.mseed'
        window.write(os.path.join(output_dir, filename), format='MSEED')

        metadata.append({
            'index': i,
            'filename': filename,
            'pick_time': pick_time.isoformat(),
            'station': window[0].stats.station,
            'network': window[0].stats.network,
            'start_time': window[0].stats.starttime.isoformat(),
            'end_time': window[0].stats.endtime.isoformat()
        })

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, metadata_filename), index=False)
