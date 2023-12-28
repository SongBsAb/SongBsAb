

from speaker_encoder.ecapa_tdnn import ecapa_tdnn
from speaker_encoder.utils import parse_enroll_model_file, parse_mean_file_2

from speaker_encoder._auto_speech.models import resnet as ResNet
from speaker_encoder._auto_speech.data_objects.audio import normalize_volume_torch, wav_to_spectrogram_torch
from speaker_encoder._auto_speech.data_objects.params_data import *

import torch
import torch.nn as nn
import numpy as np

BITS = 16

class resnet(ecapa_tdnn):

    def __init__(self, model_name, extractor_file, mean_file=None, feat_mean_file=None, feat_std_file=None, 
                model_file=None, threshold=None, device='cpu',
                ):

        nn.Module.__init__(self)

        self.device = device

        checkpoint = torch.load(extractor_file)
        MODEL_NUM_CLASSES = checkpoint['state_dict']['classifier.weight'].shape[0]
        MODEL_NAME = model_name
        self.encoder = eval('ResNet.{}(num_classes={}, pretrained=False)'.format(MODEL_NAME, MODEL_NUM_CLASSES))
        # load checkpoint
        self.encoder.load_state_dict(checkpoint['state_dict'])
        self.encoder.to(device).eval()
        assert self.encoder.training == False

        self.emb_mean = parse_mean_file_2(mean_file, device)

        # feature mean and std
        if feat_mean_file is not None:
            self.feat_mean = torch.load(feat_mean_file, map_location=self.device)
            assert self.feat_mean.shape[0] == n_fft // 2 + 1
        else:
            self.feat_mean = 0
        if feat_std_file is not None:
            self.feat_std = torch.load(feat_std_file, map_location=self.device)
            assert self.feat_std.shape[0] == n_fft // 2 + 1
        else:
            self.feat_std = 1
        
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
        x = normalize_volume_torch(x, audio_norm_target_dBFS, increase_only=True) # (batch, time)
        feats = wav_to_spectrogram_torch(x) # (batch, frames, #bin)
        return feats

    
    def cmvn(self, feats):
        feats = (feats - self.feat_mean) / self.feat_std
        return feats
    

    def extract_emb(self, x):
        emb = self.encoder(x) - self.emb_mean
        return emb