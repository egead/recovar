import numpy as np
import obspy
from obspy import UTCDateTime
import h5py
import pandas as pd
import os
from scipy.signal import detrend


class ContinuousDataPreprocessor:

    def __init__(self,
                 catalog_csv,
                 output_hdf5_path,
                 output_metadata_csv_path,
                 window_length=60,
                 sampling_rate=100,
                 freqmin=1,
                 freqmax=20):
        self.catalog_csv = catalog_csv
        self.output_hdf5_path = output_hdf5_path
        self.output_metadata_csv_path = output_metadata_csv_path
        self.window_length = window_length
        self.sampling_rate = sampling_rate
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.expected_samples = int(window_length * sampling_rate)

        self.catalog = pd.read_csv(catalog_csv)
        if 'p_arrival_time' in self.catalog.columns:
            self.catalog['p_arrival_time'] = pd.to_datetime(
                self.catalog['p_arrival_time'], format='ISO8601'
            )
        if 's_arrival_time' in self.catalog.columns:
            self.catalog['s_arrival_time'] = pd.to_datetime(
                self.catalog['s_arrival_time'], format='ISO8601'
            )

        self._catalog_by_station = {}
        for station, group in self.catalog.groupby('station'):
            events = []
            for _, row in group.iterrows():
                p_time = None
                s_time = None
                if pd.notna(row.get('p_arrival_time')):
                    p_time = UTCDateTime(row['p_arrival_time'])
                if pd.notna(row.get('s_arrival_time')):
                    s_time = UTCDateTime(row['s_arrival_time'])
                if p_time is not None:
                    events.append((p_time, s_time))
            self._catalog_by_station[station] = events

        f = np.fft.fftfreq(self.expected_samples, d=1.0 / self.sampling_rate)
        self._freq_pass = (np.abs(f) >= self.freqmin) & (np.abs(f) <= self.freqmax)

        self.trace_counter = 0

    def process_station(self, station_dir):
        try:
            stream = obspy.read(os.path.join(station_dir, "*.mseed"))
        except Exception as e:
            print(f"Error reading MSEED files in {station_dir}: {e}")
            return

        if len(stream) == 0:
            return

        for tr in stream:
            tr.data = tr.data.astype(np.float32)

        try:
            stream = stream.merge(fill_value=np.nan)
        except Exception as e:
            print(f"Error merging streams: {e}")
            return

        station_name = stream[0].stats.station
        network = stream[0].stats.network

        for tr in stream:
            if tr.stats.sampling_rate != self.sampling_rate:
                tr.resample(self.sampling_rate)

        z_trace = stream.select(channel="*Z")
        n_trace = stream.select(channel="*N")
        e_trace = stream.select(channel="*E")

        if not (z_trace and n_trace and e_trace):
            z_trace = stream.select(channel="*HZ")
            n_trace = stream.select(channel="*HN")
            e_trace = stream.select(channel="*HE")

        if not (z_trace and n_trace and e_trace):
            print(f"Missing components for station {station_name}")
            return

        z_trace = z_trace[0]
        n_trace = n_trace[0]
        e_trace = e_trace[0]

        start_time = max(z_trace.stats.starttime, n_trace.stats.starttime, e_trace.stats.starttime)
        end_time = min(z_trace.stats.endtime, n_trace.stats.endtime, e_trace.stats.endtime)

        z_trace.trim(start_time, end_time)
        n_trace.trim(start_time, end_time)
        e_trace.trim(start_time, end_time)

        z_data = np.array(z_trace.data, dtype=np.float32)
        n_data = np.array(n_trace.data, dtype=np.float32)
        e_data = np.array(e_trace.data, dtype=np.float32)

        del z_trace, n_trace, e_trace, stream

        n_samples = self.expected_samples
        total_samples = min(len(z_data), len(n_data), len(e_data))
        n_windows = (total_samples - n_samples) // n_samples + 1

        if n_windows <= 0:
            return

        station_events = self._catalog_by_station.get(station_name, [])
        station_events.sort(key=lambda x: x[0])
        p_times_float = np.array(
            [float(ev[0]) for ev in station_events], dtype=np.float64
        ) if station_events else np.array([], dtype=np.float64)

        metadata = []
        valid_windows = []
        valid_window_data = []

        start_time_float = float(start_time)
        window_length_float = float(self.window_length)

        for i in range(n_windows):
            idx_start = i * n_samples
            idx_end = idx_start + n_samples

            z_win = z_data[idx_start:idx_end]
            n_win = n_data[idx_start:idx_end]
            e_win = e_data[idx_start:idx_end]

            if not self._is_window_valid(z_win, n_win, e_win):
                continue

            z_processed = self._preprocess_trace(z_win)
            n_processed = self._preprocess_trace(n_win)
            e_processed = self._preprocess_trace(e_win)
            window_data = np.stack([e_processed, n_processed, z_processed], axis=-1)

            window_start_float = start_time_float + i * window_length_float
            window_end_float = window_start_float + window_length_float

            label, p_sample, s_sample = self._check_earthquake(
                station_events, p_times_float,
                window_start_float, window_end_float
            )

            current_time = start_time + i * self.window_length
            trace_name = f"{network}.{station_name}.{current_time.strftime('%Y%m%d_%H%M%S')}"

            valid_windows.append(trace_name)
            valid_window_data.append(window_data)

            metadata.append({
                'trace_name': trace_name,
                'station_name': station_name,
                'network': network,
                'trace_start_time': current_time.isoformat(),
                'label': label,
                'p_arrival_sample': p_sample if p_sample is not None else np.nan,
                's_arrival_sample': s_sample if s_sample is not None else np.nan,
                'source_id': trace_name if label == 'no' else f"eq_{self.trace_counter}",
                'trace_category': 'earthquake_local' if label == 'eq' else 'noise',
            })
            self.trace_counter += 1

        if valid_window_data:
            with h5py.File(self.output_hdf5_path, 'a') as h5f:
                if 'data' not in h5f:
                    data_group = h5f.create_group('data')
                else:
                    data_group = h5f['data']

                for name, data in zip(valid_windows, valid_window_data):
                    data_group.create_dataset(
                        name, data=data,
                        compression='gzip', compression_opts=1
                    )

        if metadata:
            metadata_df = pd.DataFrame(metadata)
            if os.path.exists(self.output_metadata_csv_path):
                metadata_df.to_csv(
                    self.output_metadata_csv_path, mode='a',
                    header=False, index=False
                )
            else:
                metadata_df.to_csv(self.output_metadata_csv_path, index=False)

            print(f"Processed {len(metadata)} windows from station {station_name}")

    def _is_window_valid(self, z_data, n_data, e_data):
        for win in (z_data, n_data, e_data):
            if hasattr(win, 'mask'):
                if win.mask.any():
                    return False
            if len(win) != self.expected_samples:
                return False

        stacked = np.stack([z_data, n_data, e_data])
        if not np.isfinite(stacked).all():
            return False
        if np.any(np.all(stacked == 0, axis=1)):
            return False
        return True

    def _preprocess_trace(self, data):
        data = data - np.mean(data)
        data = detrend(data, type='linear')

        xw = np.fft.fft(data)
        xw[~self._freq_pass] = 0
        filtered = np.real(np.fft.ifft(xw)).astype(np.float32)

        return filtered

    def _check_earthquake(self, station_events, p_times_float,
                          window_start, window_end):
        if len(p_times_float) == 0:
            return 'no', None, None

        left = np.searchsorted(p_times_float, window_start, side='left')
        right = np.searchsorted(p_times_float, window_end, side='right')

        for idx in range(left, right):
            p_time, s_time = station_events[idx]
            p_sample = int((float(p_time) - window_start) * self.sampling_rate)

            s_sample = None
            if s_time is not None:
                s_float = float(s_time)
                if window_start <= s_float <= window_end:
                    s_sample = int((s_float - window_start) * self.sampling_rate)

            return 'eq', p_sample, s_sample

        return 'no', None, None
