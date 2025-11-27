import numpy as np
from scipy.signal import detrend
from recovar.classifier_models import ClassifierMultipleAutoencoder
from recovar.representation_learning_models import RepresentationLearningMultipleAutoencoder

def load_recovar_classifier(MODEL_PATH):
    model = RepresentationLearningMultipleAutoencoder()
    model.compile()
    model(np.random.randn(1, 3000, 3).astype(np.float32))
    model.load_weights(MODEL_PATH)
    classifier = ClassifierMultipleAutoencoder(model)

    return classifier

def preprocess(data, sampling_rate=100.0, freqmin=1.0, freqmax=20.0):
    f = np.fft.fftfreq(len(data), d=1/sampling_rate)
    data = data - np.mean(data)
    data = detrend(data, type='linear')
    xw = np.fft.fft(data)
    mask = (np.abs(f) < freqmin) | (np.abs(f) > freqmax)
    xw[mask] = 0
    return np.real(np.fft.ifft(xw)).astype(np.float32)

def validate_stream(stream, sampling_rate=100.0, channel_pattern="H*", window_size=3000, trim_samples=500):
    """
    Validates that the incoming stream is in the correct format for RECOVAR processing.

    Requirements:
    - All channels have the same length
    - Sampling rate matches expected value
    - Data length is at least window_size + 2*trim_samples
    - Channel pattern matches expected pattern (E, N, Z components)
    - No gaps in the stream

    :param stream: ObsPy Stream object to validate
    :param sampling_rate: Expected sampling rate in Hz (default: 100.0)
    :param channel_pattern: Channel pattern prefix (default: "H*" for H*E, H*N, H*Z)
    :param window_size: Window size in samples (default: 3000)
    :param trim_samples: Samples to trim from each end (default: 500)
    :raises ValueError: If any validation check fails
    """
    if len(stream) < 3:
        raise ValueError(f"Stream must have at least 3 traces (E, N, Z), found {len(stream)}")

    try:
        z_trace = stream.select(channel=f"{channel_pattern}Z")[0]
        n_trace = stream.select(channel=f"{channel_pattern}N")[0]
        e_trace = stream.select(channel=f"{channel_pattern}E")[0]
    except IndexError:
        channels_present = [tr.stats.channel for tr in stream]
        raise ValueError(f"Stream must contain {channel_pattern}E, {channel_pattern}N, and {channel_pattern}Z channels. Found: {channels_present}")

    if stream.get_gaps():
        raise ValueError("Stream has gaps")

    for tr in stream:
        if abs(tr.stats.sampling_rate - sampling_rate) > 0.01:
            raise ValueError(f"Sampling rate must be {sampling_rate} Hz, found {tr.stats.sampling_rate} Hz for {tr.id}")

    lengths = [len(tr.data) for tr in stream]
    if len(set(lengths)) > 1:
        raise ValueError(f"All channels must have the same length. Found lengths: {lengths}")

    duration_samples = lengths[0]
    min_samples = window_size + 2 * trim_samples
    min_duration_seconds = min_samples / sampling_rate

    if duration_samples < min_samples:
        actual_duration = duration_samples / sampling_rate
        raise ValueError(
            f"Data must be at least {min_samples} samples ({min_duration_seconds:.1f} seconds) "
            f"to support window_size={window_size} and trim_samples={trim_samples}. "
            f"Found {duration_samples} samples ({actual_duration:.1f} seconds)"
        )

    for tr in stream:
        if np.isnan(tr.data).any():
            raise ValueError(f"NaN values found in trace {tr.id}")

def recovar_pick_cleaner_sliding(stream, classifier, window_size=3000, stride=100,
                                  trim_samples=500, sampling_rate=100.0, channel_pattern="H*",
                                  freqmin=1.0, freqmax=20.0):
    """
    Applies RECOVAR classifier to a obspy.Stream using sliding windows.

    Requirements are validated with validate_stream() function.

    :param stream: obspy.Stream object with 3 components (E, N, Z)
    :param classifier: RECOVAR classifier instance (ClassifierMultipleAutoencoder)
    :param window_size: Window size in samples (default: 3000 = 30 seconds at 100 Hz)
    :param stride: Stride in samples (default: 100 = 1 second at 100 Hz)
    :param trim_samples: Samples to trim from each end after preprocessing (default: 500 = 5 seconds)
    :param sampling_rate: Expected sampling rate in Hz (default: 100.0)
    :param channel_pattern: Channel pattern prefix (default: "H*")
    :param freqmin: Minimum frequency for bandpass filter in Hz (default: 1.0)
    :param freqmax: Maximum frequency for bandpass filter in Hz (default: 20.0)
    :return: Dictionary with 'scores_array' (np.array), 'mean_score' (float), 'max_score' (float)
    """

    validate_stream(stream, sampling_rate=sampling_rate, channel_pattern=channel_pattern,
                    window_size=window_size, trim_samples=trim_samples)

    for tr in stream:
        tr.data = tr.data.astype(np.float32)

    z_trace = stream.select(channel=f"{channel_pattern}Z")[0]
    n_trace = stream.select(channel=f"{channel_pattern}N")[0]
    e_trace = stream.select(channel=f"{channel_pattern}E")[0]

    e_processed = preprocess(e_trace.data, sampling_rate=sampling_rate, freqmin=freqmin, freqmax=freqmax)
    n_processed = preprocess(n_trace.data, sampling_rate=sampling_rate, freqmin=freqmin, freqmax=freqmax)
    z_processed = preprocess(z_trace.data, sampling_rate=sampling_rate, freqmin=freqmin, freqmax=freqmax)

    e_trimmed = e_processed[trim_samples:-trim_samples]
    n_trimmed = n_processed[trim_samples:-trim_samples]
    z_trimmed = z_processed[trim_samples:-trim_samples]

    n_windows = (len(e_trimmed) - window_size) // stride + 1

    waveforms = []
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size

        e_window = e_trimmed[start_idx:end_idx]
        n_window = n_trimmed[start_idx:end_idx]
        z_window = z_trimmed[start_idx:end_idx]

        waveform = np.stack([e_window, n_window, z_window], axis=-1)
        waveforms.append(waveform)

    waveforms = np.array(waveforms)

    scores = classifier(waveforms)

    return {
        'scores_array': scores,
        'mean_score': np.mean(scores),
        'max_score': np.max(scores)
    }


def recovar_pick_cleaner_sliding_batch(streams, classifier, batch_size=256, window_size=3000,
                                        stride=100, trim_samples=500, sampling_rate=100.0,
                                        channel_pattern="H*", freqmin=1.0, freqmax=20.0):
    """
    Applies RECOVAR classifier to multiple obspy.Stream objects using sliding windows in batches.

    :param streams: List of obspy.Stream objects with 3 components (E, N, Z) each
    :param classifier: RECOVAR classifier instance (ClassifierMultipleAutoencoder)
    :param batch_size: Number of windows to process at once (default: 256)
    :param window_size: Window size in samples (default: 3000 = 30 seconds at 100 Hz)
    :param stride: Stride in samples (default: 100 = 1 second at 100 Hz)
    :param trim_samples: Samples to trim from each end after preprocessing (default: 500 = 5 seconds)
    :param sampling_rate: Expected sampling rate in Hz (default: 100.0)
    :param channel_pattern: Channel pattern prefix (default: "H*")
    :param freqmin: Minimum frequency for bandpass filter in Hz (default: 1.0)
    :param freqmax: Maximum frequency for bandpass filter in Hz (default: 20.0)
    :return: List of dictionaries with 'scores_array', 'mean_score', 'max_score' for each stream
    """
    all_waveforms = []
    window_counts = []

    for stream in streams:
        validate_stream(stream, sampling_rate=sampling_rate, channel_pattern=channel_pattern,
                    window_size=window_size, trim_samples=trim_samples)

        for tr in stream:
            tr.data = tr.data.astype(np.float32)

        z_trace = stream.select(channel=f"{channel_pattern}Z")[0]
        n_trace = stream.select(channel=f"{channel_pattern}N")[0]
        e_trace = stream.select(channel=f"{channel_pattern}E")[0]

        e_processed = preprocess(e_trace.data, sampling_rate=sampling_rate, freqmin=freqmin, freqmax=freqmax)
        n_processed = preprocess(n_trace.data, sampling_rate=sampling_rate, freqmin=freqmin, freqmax=freqmax)
        z_processed = preprocess(z_trace.data, sampling_rate=sampling_rate, freqmin=freqmin, freqmax=freqmax)

        e_trimmed = e_processed[trim_samples:-trim_samples]
        n_trimmed = n_processed[trim_samples:-trim_samples]
        z_trimmed = z_processed[trim_samples:-trim_samples] 

        n_windows = (len(e_trimmed) - window_size) // stride + 1
        window_counts.append(n_windows)

        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size

            e_window = e_trimmed[start_idx:end_idx]
            n_window = n_trimmed[start_idx:end_idx]
            z_window = z_trimmed[start_idx:end_idx]

            waveform = np.stack([e_window, n_window, z_window], axis=-1)
            all_waveforms.append(waveform)

    all_waveforms = np.array(all_waveforms)
    n_samples = len(all_waveforms)
    n_complete_batches = n_samples // batch_size
    remainder = n_samples % batch_size

    all_scores = []

    for i in range(n_complete_batches):
        batch = all_waveforms[i * batch_size:(i + 1) * batch_size]
        batch_results = classifier(batch)
        all_scores.extend(batch_results)

    if remainder > 0:
        last_batch = all_waveforms[n_complete_batches * batch_size:]
        padding = np.zeros((batch_size - remainder, window_size, 3), dtype=np.float32)
        padded_batch = np.concatenate([last_batch, padding], axis=0)
        batch_results = classifier(padded_batch)
        all_scores.extend(batch_results[:remainder])

    all_scores = np.array(all_scores)

    results = []
    start_idx = 0
    for n_windows in window_counts:
        end_idx = start_idx + n_windows
        stream_scores = all_scores[start_idx:end_idx]

        results.append({
            'scores_array': stream_scores,
            'mean_score': np.mean(stream_scores),
            'max_score': np.max(stream_scores)
        })

        start_idx = end_idx

    return results
