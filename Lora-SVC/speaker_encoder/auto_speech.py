
import numpy as np
import torch
import torch.nn as nn

from speaker_encoder.resnet import resnet
from speaker_encoder._auto_speech.models.model import Network
from speaker_encoder._auto_speech.data_objects.params_data import *
from speaker_encoder.utils import parse_enroll_model_file, parse_mean_file_2

MODEL_LAYERS = 8 # same for iden and veri
MODEL_INIT_CHANNELS = 128 # same for iden and veri

from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
text_arch = "Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), \
    ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), \
        ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('max_pool_3x3', 1)], \
            normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), \
                ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), \
                    ('dil_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 2)], \
                        reduce_concat=range(2, 6))" # the best arch reported in the paper
# "Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))"
genotype = eval(text_arch)


class auto_speech(resnet):

    def __init__(self, extractor_file, mean_file=None, feat_mean_file=None, feat_std_file=None, 
                model_file=None, threshold=None, device='cpu',
                ):
        
        nn.Module.__init__(self)

        self.device = device

        checkpoint = torch.load(extractor_file)
        MODEL_NUM_CLASSES = checkpoint['state_dict']['classifier.weight'].shape[0]
        # genotype = checkpoint['genotype']
        self.encoder = Network(MODEL_INIT_CHANNELS, MODEL_NUM_CLASSES, MODEL_LAYERS, genotype)
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