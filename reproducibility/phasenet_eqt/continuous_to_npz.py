import numpy as np
import obspy
from scipy.signal import detrend
from pathlib import Path

def process_mseed_to_npy(mseed_file, output_dir, sampling_rate=100, freqmin=1.0, freqmax=20.0):
    print(f"Reading: {mseed_file}")
    stream = obspy.read(mseed_file)

    for tr in stream:
        tr.data = tr.data.astype(np.float32)

    stream = stream.merge(fill_value=np.nan)

    for tr in stream:
        if tr.stats.sampling_rate != sampling_rate:
            tr.resample(sampling_rate)

    z_trace = stream.select(channel="HHZ")# or stream.select(channel="*HZ")
    n_trace = stream.select(channel="HHN")# or stream.select(channel="*HN")
    e_trace = stream.select(channel="HHE")# or stream.select(channel="*HE")

    if not (z_trace and n_trace and e_trace):
        print(f"Warning: Missing E, N, or Z components in {mseed_file}, skipping...")
        return None

    z_trace = z_trace[0]
    n_trace = n_trace[0]
    e_trace = e_trace[0]

    start_time = max(z_trace.stats.starttime, n_trace.stats.starttime, e_trace.stats.starttime)
    end_time = min(z_trace.stats.endtime, n_trace.stats.endtime, e_trace.stats.endtime)

    z_trace.trim(start_time, end_time)
    n_trace.trim(start_time, end_time)
    e_trace.trim(start_time, end_time)

    print(f"Time: {start_time} to {end_time}")

    f = np.fft.fftfreq(len(z_trace.data), d=1/sampling_rate)

    def preprocess(data):
        data = data - np.mean(data)
        data = detrend(data, type='linear')
        xw = np.fft.fft(data)
        mask = (np.abs(f) < freqmin) | (np.abs(f) > freqmax)
        xw[mask] = 0
        return np.real(np.fft.ifft(xw)).astype(np.float32)

    print("Preprocessing...")
    e_processed = preprocess(e_trace.data)
    n_processed = preprocess(n_trace.data)
    z_processed = preprocess(z_trace.data)

    waveform = np.stack([e_processed, n_processed, z_processed], axis=-1)

    # Create output filename using station name
    station_name = z_trace.stats.station
    network_name = z_trace.stats.network
    output_filename = f"{network_name}_{station_name}.npy"
    output_path = Path(output_dir) / output_filename

    print(f"Shape: {waveform.shape}")
    print(f"Saving: {output_path}")

    np.save(output_path, waveform)

    print(f"Done: {output_filename}")
    return waveform


def batch_process_mseed_files(mseed_dir, output_dir, sampling_rate=100, freqmin=1.0, freqmax=20.0):
    """Process all mseed files in a directory and save as separate npy files."""
    mseed_dir = Path(mseed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mseed_files = list(mseed_dir.glob("*.mseed"))
    print(f"Found {len(mseed_files)} mseed files")

    successful = 0
    failed = 0

    for mseed_file in mseed_files:
        try:
            result = process_mseed_to_npy(mseed_file, output_dir, sampling_rate, freqmin, freqmax)
            if result is not None:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {mseed_file}: {e}")
            failed += 1

    print(f"\nProcessing complete: {successful} successful, {failed} failed")



MSEED_DIR = '/home/ege/2OCT_14-30/'
OUTPUT_DIR = '/home/ege/2OCT_14-30/'

batch_process_mseed_files(MSEED_DIR, OUTPUT_DIR)
