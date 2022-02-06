'''Need install
conda install -c conda-forge ffmpeg
'''

import pip

try:
    __import__('librosa')
except ImportError:
    pip.main(['install', 'librosa'])
    
import librosa
import numpy as np

def statistic_values(data, track_id, name, values):
    '''Converte os valores para cada canal em dados estatisticos e seta na tabela'''
    data.loc[track_id, (name, "mean")] = np.mean(values, axis=1)
    data.loc[track_id, (name, "std")] = np.std(values, axis=1)
    data.loc[track_id, (name, "median")] = np.median(values, axis=1)
    data.loc[track_id, (name, "min")] = np.min(values, axis=1)
    data.loc[track_id, (name, "max")] = np.max(values, axis=1)
    
def extract_features(data, track_id, path):
    '''extrai dados de audio e seta na tabela'''
    x, sr = librosa.load(path, sr=None, mono=True)
    
    statistic_values(data, track_id, 'zcr', librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512))
    
    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
    assert cqt.shape[0] == 7 * 12
    assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    statistic_values(data, track_id, 'chroma_cqt', f)
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    statistic_values(data, track_id, 'chroma_cens', f)
    f = librosa.feature.tonnetz(chroma=f)
    statistic_values(data, track_id, 'tonnetz', f)

    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
    del x

    f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    statistic_values(data, track_id, 'chroma_stft', f)

    f = librosa.feature.rms(S=stft)
    statistic_values(data, track_id, 'rmse', f)

    f = librosa.feature.spectral_centroid(S=stft)
    statistic_values(data, track_id, 'spectral_centroid', f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    statistic_values(data, track_id, 'spectral_bandwidth', f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    statistic_values(data, track_id, 'spectral_contrast', f)
    f = librosa.feature.spectral_rolloff(S=stft)
    statistic_values(data, track_id, 'spectral_rolloff', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    statistic_values(data, track_id, 'mfcc', f)