from glob import glob

import librosa
import theano
import numpy as np
import os

from tqdm import tqdm

floatX = theano.config.floatX
GENRES = ["blues", "classical", "country", "disco",
          "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


def load_audio_get_spectogram(audio_file, offset=0.0, duration=None):
    x, sr = librosa.load(audio_file, offset=offset,
                         duration=duration, dtype=floatX)
    D = librosa.stft(x)
    S = np.log1p(np.abs(D)).astype(floatX)
    return S


def get_spectogram_samples(audio_file, sample_len_secs, n_samples):
    x, sr = librosa.load(audio_file, dtype=floatX)
    sample_len_frames = sr * sample_len_secs
    sample_frames = [x[i:i + sample_len_frames]
                     for i in [np.random.randint(0, len(x) - sample_len_frames) for _ in range(n_samples)]]
    sample_Ds = [librosa.stft(s) for s in sample_frames]
    sample_Ss = [np.log1p(np.abs(D)).astype(floatX) for D in sample_Ds]
    sample_Ss = [S.reshape(S.shape[0], 1, S.shape[1]) for S in sample_Ss]
    return np.stack(sample_Ss)


def load_dataset(root_path, sample_len_secs, n_samples_per_file):
    Xs, ys = [], []
    for i, genre in enumerate(GENRES):
        for audio_file in tqdm(list(glob(os.path.join(root_path, genre, "*.au"))), ncols=80, ascii=False, desc="Loading {} files".format(genre)):
            Xs.extend(get_spectogram_samples(
                audio_file, sample_len_secs, n_samples_per_file))
            ys.extend([i for _ in range(n_samples_per_file)])
    return np.stack(Xs), np.stack(ys)
