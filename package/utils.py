
from typing import Iterable, Mapping, Tuple, Optional

import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import DatasetDict
from scipy import signal


def iterate_over_all_samples(dataset: DatasetDict, show_progress_bar=False) -> Iterable[Mapping]:

    total = sum([d.num_rows for d in dataset.values()])
    with tqdm(total=total, disable=not show_progress_bar) as pbar:
        for ds_name, data in dataset.items():
            for row in tqdm(data):
                pbar.update(1)
                row['ds_name'] = ds_name
                yield row


def log_specgram(audio: np.ndarray, sample_rate: int, window_size=20, step_size=10, eps=1e-10) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)

    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def plot_spectrogram(samples: np.ndarray, sample_rate: int, filename: str):

    freqs, times, spectrogram = log_specgram(samples, sample_rate)

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of ' + filename)
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sample_rate / len(samples), sample_rate), samples)

    ax2 = fig.add_subplot(212)
    ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax2.set_yticks(freqs[::16])
    ax2.set_xticks(times[::16])
    ax2.set_title('Spectrogram of ' + filename)
    ax2.set_ylabel('Freqs in Hz')
    ax2.set_xlabel('Seconds')


def pad_audio(y: np.ndarray, duration=16000, clip_silence=True, top_db=60) -> np.ndarray:

    # Clip silence
    yt = librosa.effects.trim(y, top_db=top_db) if clip_silence else y

    # Pad to a length
    if len(yt) > duration:
        yt = yt[:duration]
    else:
        padding = duration - len(yt)
        offset = padding // 2
        yt = np.pad(yt, (offset, duration - len(yt) - offset), 'constant')

    return yt


def create_melspec(y: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:

    melspec = librosa.feature.melspectrogram(y,
                                             sr=sr,
                                             n_mels=n_mels,
                                             fmax=(sr // 2))
    melspec_b = librosa.power_to_db(melspec, ref=np.max)
    melspec_b = melspec_b.astype(np.float32)

    return melspec_b


def mono_to_color(X, mean: Optional[float] = None, std: Optional[float] = None, norm_max: Optional[float] = None,
                  norm_min: Optional[float] = None, eps=1e-6) -> np.ndarray:

    X = np.stack([X, X, X], axis=-1)
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V
