
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

from speaker_encoder.utils import check_input_range, parse_enroll_model_file, parse_mean_file_2
from speaker_encoder.xv_plda import xv_plda

BITS = 16

n_mels = 80
left_frames = 0
right_frames = 0
deltas = False
# fbank = Fbank(deltas=deltas, n_mels=n_mels, left_frames=left_frames, right_frames=right_frames)

norm_type = 'sentence'
std_norm = False
# inorm = InputNormalization(norm_type=norm_type, std_norm=std_norm)

input_size = n_mels
channels = [1024, 1024, 1024, 1024, 3072]
kernel_sizes = [5, 3, 3, 3, 1]
dilations = [1, 2, 3, 4, 1]
attention_channels = 128
lin_neurons = 192


class ecapa_tdnn(xv_plda):

    def __init__(self, extractor_file, mean_file=None, model_file=None, threshold=None, device='cpu'):
        
        nn.Module.__init__(self)

        self.device = device

        self.encoder = ECAPA_TDNN(input_size=input_size, channels=channels, 
                                    kernel_sizes=kernel_sizes, dilations=dilations, 
                                    attention_channels=attention_channels, lin_neurons=lin_neurons,
                                    device=device)
        state_dict = torch.load(extractor_file, map_location=device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval().to(self.device)

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
    

    def raw(self, x):
        """
        x: (B, 1, T)
        """
        x = x.squeeze(1)
        feats = self.fbank(x)
        return feats

    
    def cmvn(self, feats):
        lens = torch.ones(feats.shape[0], device=feats.device)
        feats = self.inorm(feats, lens)
        return feats

        
    def scoring_trials(self, enroll_embs, embs):

        return F.cosine_similarity(embs.unsqueeze(2), enroll_embs.unsqueeze(0).transpose(1, 2))
    

    def extract_emb(self, x):

        emb = self.encoder(x).squeeze(1) - self.emb_mean
        return emb
    