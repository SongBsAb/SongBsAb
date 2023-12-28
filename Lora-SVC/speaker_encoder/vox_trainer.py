
#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import importlib
import numpy as np

import torch.nn.functional as F

class SpeakerNet(nn.Module):
    def __init__(self, model, device='cpu', checkpoint=None, **kwargs):
        super(SpeakerNet, self).__init__()

        self.device = device
        
        SpeakerNetModel = importlib.import_module("speaker_encoder._vox_trainer." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location='cpu')
            ckpt_new = {}
            for key, parameter in ckpt.items():
                if '__S__' not in key:
                    continue
                ckpt_new[key] = parameter
            self.load_state_dict(ckpt_new)
        
        self.__S__.to(self.device).eval()

    def forward_2(self, data):
        '''
        data: (bs, 1, sample)
        '''
        if len(data.shape) == 3:
            assert data.shape[1] == 1
            data = data.squeeze(1)
        data = data.to(self.device)
        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.__S__.forward(data)

        return outp
    
    def embedding(self, data, partial_utterance_n_frames=None):
        if partial_utterance_n_frames is not None:
            x = data
            wav_slices, _ = self.compute_partial_slices(x.shape[-1])
            embs = []
            for slice in wav_slices:
                start = slice.start
                end = slice.stop
                x_part = x[..., start:end]
                emb_part = self.forward_2(x_part) # (B, D)
                embs.append(emb_part)
            embs = torch.stack(embs).transpose(0, 1)
            return embs
        else:
            return self.forward_2(data)
    
    def compute_partial_slices(self, n_samples, partial_utterance_n_frames=160,
                            min_pad_coverage=0.75, overlap=0.5, mel_window_step=10, sampling_rate=16_000):
        
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
    
    def forward(self, x, flag=0, return_emb=False, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        assert flag == 0
        embedding = self.embedding(x)
        # print('############## Nan Check:', np.any(np.isnan(embedding.detach().cpu().numpy())))
        
        if not hasattr(self, 'enroll_embs'):
            assert enroll_embs is not None
        enroll_embs = enroll_embs if enroll_embs is not None else self.enroll_embs
        scores = self.scoring_trials(enroll_embs=enroll_embs, embs=embedding)
        # scores = torch.nn.functional.cosine_similarity(embedding.unsqueeze(2), enroll_embs.transpose(0, 1).unsqueeze(0), dim=1)
        if not return_emb:
            return scores
        else:
            return scores, embedding
    
    def scoring_trials(self, enroll_embs, embs):

        return F.cosine_similarity(embs.unsqueeze(2), enroll_embs.unsqueeze(0).transpose(1, 2))