from glob import glob

import librosa
import theano
import numpy as np
import os

from tqdm import tqdm, trange
from sklearn.externals import joblib

floatX = theano.config.floatX
GENRES = ["blues", "classical", "country", "disco",
          "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
SR = 22050
N_FFT = 256


def load_audio_get_spectrogram(audio_file, offset=0.0, duration=None):
    x, sr = librosa.load(audio_file, offset=offset,
                         duration=duration, dtype=floatX, sr=SR)
    D = librosa.stft(x, n_fft=N_FFT)
    S = np.log1p(np.abs(D)).astype(floatX)
    return S


def get_spectrogram_samples(audio_file, sample_len_secs, n_samples):
    x, sr = librosa.load(audio_file, dtype=floatX, sr=SR)
    sample_len_frames = sr * sample_len_secs
    sample_frames = [x[i:i + sample_len_frames]
                     for i in [np.random.randint(0, len(x) - sample_len_frames) for _ in range(n_samples)]]
    sample_Ds = [librosa.stft(s, n_fft=N_FFT) for s in sample_frames]
    sample_Ss = [np.log1p(np.abs(D)).astype(floatX) for D in sample_Ds]
    sample_Ss = [S.reshape(1, S.shape[0], S.shape[1]) for S in sample_Ss]
    return np.stack(sample_Ss)


def load_dataset(root_path, sample_len_secs, n_samples_per_file):
    Xs, ys = [], []
    for i, genre in enumerate(GENRES):
        for audio_file in tqdm(list(glob(os.path.join(root_path, genre, "*.au"))), ncols=80, ascii=False, desc="Loading {} files".format(genre)):
            Xs.extend(get_spectrogram_samples(
                audio_file, sample_len_secs, n_samples_per_file))
            ys.extend([i for _ in range(n_samples_per_file)])
    X, y = np.stack(Xs), np.stack(ys)
    joblib.dump([X, y], os.path.join("processed_data", "xy_{}s_{}.pkl".format(sample_len_secs, n_samples_per_file)))
    return X, y


def convert_spectrogram_and_save(S, output_file):
    x = np.exp(S) - 1
    p = 2 * np.pi * np.random.random_sample(x.shape) - np.pi
    for i in trange(500, desc="Inverting spectrogram", ncols=80):
        Q = x * np.exp(1j*p)
        y = librosa.istft(Q) + 1e-5
        p = np.angle(librosa.stft(y, n_fft=N_FFT))
    librosa.output.write_wav(output_file, y, SR)

