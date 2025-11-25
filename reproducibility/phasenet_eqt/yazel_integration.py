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

def preprocess(data,sampling_rate=100.0,freqmin=1.0, freqmax=20.0):
    f = np.fft.fftfreq(len(data), d=1/sampling_rate)
    data = data - np.mean(data)
    data = detrend(data, type='linear')
    xw = np.fft.fft(data)
    mask = (np.abs(f) < freqmin) | (np.abs(f) > freqmax)
    xw[mask] = 0
    return np.real(np.fft.ifft(xw)).astype(np.float32)


def recovar_pick_cleaner(stream, pick_idx, classifier, threshold=None):

    for tr in stream: 
        tr.data = tr.data.astype(np.float32)

    z_trace = stream.select(channel="HHZ")[0]
    n_trace = stream.select(channel="HHN")[0]
    e_trace = stream.select(channel="HHE")[0]

    if stream.get_gaps():
        raise ValueError("Stream has gaps")

    for tr in stream:
        if np.isnan(tr.data).any():
            raise ValueError(f"NaN values in trace {tr.id}")

    e_processed = preprocess(e_trace.data)
    n_processed = preprocess(n_trace.data)
    z_processed = preprocess(z_trace.data)

    e_processed = e_processed[len(e_processed) // 2 - 1500 : len(e_processed) // 2 + 1500]
    n_processed = n_processed[len(n_processed) // 2  - 1500 : len(n_processed) // 2  + 1500]
    z_processed = z_processed[len(z_processed) // 2  - 1500 : len(z_processed) // 2  + 1500]

    waveform = np.stack([e_processed, n_processed, z_processed], axis=-1)
    waveform = np.expand_dims(waveform, axis=0)

    return classifier(waveform)






