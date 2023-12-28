


import torch
import torch.nn as nn
import numpy as np
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.Xvector import Xvector

from speaker_encoder.ecapa_tdnn import ecapa_tdnn
from speaker_encoder.utils import parse_enroll_model_file, parse_mean_file_2

BITS = 16

n_mels = 24
left_frames = 0
right_frames = 0
deltas = False
# fbank = Fbank(deltas=deltas, n_mels=n_mels, left_frames=left_frames, right_frames=right_frames)

norm_type = 'sentence'
std_norm = False
# inorm = InputNormalization(norm_type=norm_type, std_norm=std_norm)

in_channels = n_mels
activation = torch.nn.LeakyReLU
tdnn_blocks = 5
tdnn_channels = [512, 512, 512, 512, 1500]
tdnn_kernel_sizes = [5, 3, 3, 1, 1]
tdnn_dilations = [1, 2, 3, 1, 1]
lin_neurons = 512


class xv_coss(ecapa_tdnn):

    def __init__(self, extractor_file, mean_file=None, model_file=None, threshold=None, device='cpu'):

        nn.Module.__init__(self)

        self.device = device

        self.encoder = Xvector(device, activation, tdnn_blocks, 
                        tdnn_channels, tdnn_kernel_sizes, tdnn_dilations, 
                        lin_neurons, in_channels)
        state_dict = torch.load(extractor_file, map_location=device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval().to(device)

        self.fbank = Fbank(deltas=deltas, n_mels=n_mels, left_frames=left_frames, right_frames=right_frames)
        self.inorm = InputNormalization(norm_type=norm_type, std_norm=std_norm)

        self.emb_mean = parse_mean_file_2(mean_file, device)

        if model_file is not None:
            self.num_spks, self.spk_ids, self.z_norm_means, self.z_norm_stds, self.enroll_embs = \
                parse_enroll_model_file(model_file, self.device)

        # If you need SV or OSI, must input threshold
        self.threshold = threshold if threshold else -np.infty # Valid for SV and OSI tasks; CSI: -infty

        self.allowed_flags = sorted([
            0, 1, 2
        ])
        self.range_type = 'scale'
    

    def compute_partial_slices(self, n_samples, partial_utterance_n_frames=160,
                           min_pad_coverage=0.75, overlap=0.5, sampling_rate=16_000, mel_window_step=10):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
        partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel
        spectrogram slices are returned, so as to make each partial utterance waveform correspond to
        its spectrogram. This function assumes that the mel spectrogram parameters used are those
        defined in params_data.py.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wave_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial
        utterance
        :param min_pad_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present,
        then the last partial utterance will be considered, as if we padded the audio. Otherwise,
        it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial
        utterance, this parameter is ignored so that the function always returns at least 1 slice.
        :param overlap: by how much the partial utterance should overlap. If set to 0, the partial
        utterances are entirely disjoint.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        assert 0 <= overlap < 1
        assert 0 < min_pad_coverage <= 1

        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partial_utterance_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_pad_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def partial_embedding(self, x):

        print('Enter partial EMB')

        x_vad = x.squeeze(1)

        wave_slices, mel_slices = self.compute_partial_slices(x_vad.shape[1])
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= x_vad.shape[1]:
            # wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
            x_vad = torch.nn.functional.pad(x_vad, (0, max_wave_length - x_vad.shape[1]), mode="constant")
        
        frames = self.compute_feat(x_vad.unsqueeze(1), flag=self.allowed_flags[-1]) # (B, T, F)
        frames_batch = None
        for s in mel_slices:
            f = frames[:, s, :].unsqueeze(1)
            if frames_batch is None:
                frames_batch = f
            else:
                frames_batch = torch.cat((frames_batch, f), dim=1)
        # frames_batch # (B, n, T1, F)

        x = frames_batch
        B, n, T1, F = x.shape
        frames_batch = x.view(-1, T1, F) # (B * n , T1, F)
        partial_embeds = self.embedding(frames_batch, flag=self.allowed_flags[-1]) # (B * n, D)
        partial_embeds = partial_embeds.view(B, n, -1) # (B, n, D)
        # Compute the utterance embedding from the partial embeddings
        # raw_embed = np.mean(partial_embeds, axis=0)
        raw_embed = torch.mean(partial_embeds, axis=1) # (B, D)

        # # embed = raw_embed / np.linalg.norm(raw_embed, 2)
        # embed = raw_embed / torch.norm(raw_embed, p=2, dim=1, keepdim=True)
        # # return embed # (B, D)
        # return embed - self.emb_mean # (B, D)

        return raw_embed