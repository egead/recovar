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


def recovar_pick_cleaner_sliding(stream, classifier):
    for tr in stream:
        tr.data = tr.data.astype(np.float32)

    z_trace = stream.select(channel="H*Z")[0]
    n_trace = stream.select(channel="H*N")[0]
    e_trace = stream.select(channel="H*E")[0]

    if stream.get_gaps():
        raise ValueError("Stream has gaps")

    for tr in stream:
        if np.isnan(tr.data).any():
            raise ValueError(f"NaN values in trace {tr.id}")

    e_processed = preprocess(e_trace.data)
    n_processed = preprocess(n_trace.data)
    z_processed = preprocess(z_trace.data)

    # Trim 5 seconds from each end
    # Expected input: 7000 samples -> After trim: 6000 samples
    trim_samples = 500
    e_trimmed = e_processed[trim_samples:-trim_samples]
    n_trimmed = n_processed[trim_samples:-trim_samples]
    z_trimmed = z_processed[trim_samples:-trim_samples]

    window_size = 3000  
    stride = 100 

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


def recovar_pick_cleaner_sliding_batch(streams, classifier, batch_size=256):
    all_waveforms = []
    window_counts = []

    for stream in streams:
        for tr in stream:
            tr.data = tr.data.astype(np.float32)

        z_trace = stream.select(channel="H*Z")[0]
        n_trace = stream.select(channel="H*N")[0]
        e_trace = stream.select(channel="H*E")[0]

        if stream.get_gaps():
            raise ValueError("Stream has gaps")

        for tr in stream:
            if np.isnan(tr.data).any():
                raise ValueError(f"NaN values in trace {tr.id}")

        e_processed = preprocess(e_trace.data)
        n_processed = preprocess(n_trace.data)
        z_processed = preprocess(z_trace.data)

        trim_samples = 500
        e_trimmed = e_processed[trim_samples:-trim_samples]
        n_trimmed = n_processed[trim_samples:-trim_samples]
        z_trimmed = z_processed[trim_samples:-trim_samples]

        window_size = 3000 
        stride = 100 

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
        padding = np.zeros((batch_size - remainder, 3000, 3), dtype=np.float32)
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
