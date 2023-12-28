from ..data_objects.params_data import *
from pathlib import Path
from typing import Optional, Union
import numpy as np
import librosa

import torch

int16_max = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)

    return wav


def wav_to_spectrogram(wav):
    frames = np.abs(librosa.core.stft(
        wav,
        n_fft=n_fft,
        hop_length=int(sampling_rate * window_step / 1000),
        win_length=int(sampling_rate * window_length / 1000),
    ))
    return frames.astype(np.float32).T

def wav_to_spectrogram_torch(wav):
    ''' wav: (t, ) or (batch, t)
    '''
    reshape = False
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
        reshape = True
    frames = torch.abs(torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=int(sampling_rate * window_step / 1000),
        win_length=int(sampling_rate * window_length / 1000),
        window=torch.hann_window(window_length=int(sampling_rate * window_length / 1000)).to(wav.device),
        return_complex=True
        )).transpose(1, 2)
    if reshape:
        frames = frames.squeeze(0)
    return frames


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))

def normalize_volume_torch(wav, target_dBFS, increase_only=False, decrease_only=False):
    ''' wav: (t, ) or (batch, t)
    '''
    reshape = False
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
        reshape = True
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = torch.sqrt(torch.mean((wav * int16_max) ** 2, dim=1))
    wave_dBFS = 20 * torch.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    normalized_wav = wav.clone()
    for ii in range(wav.shape[0]):
        if dBFS_change[ii] < 0 and increase_only or dBFS_change[ii] > 0 and decrease_only:
            continue
        else:
            normalized_wav[ii] = wav[ii] * (10 ** (dBFS_change[ii] / 20))
    if reshape:
        normalized_wav = normalized_wav.view(-1)
    return normalized_wav