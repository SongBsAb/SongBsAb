

import torch.nn as nn
import torch.nn.functional as F
import os

from hubert.inference import load_model, pred_vec_t, pred_vec_t_batch

class HubertVec(nn.Module):

    def __init__(self, checkpoint_root="hubert_pretrain", checkpoint_name="hubert-soft-0d54a1f4.pt"):
        super(HubertVec, self).__init__()

        self.hubert = load_model(os.path.join(checkpoint_root, checkpoint_name), device='cuda')

    # def forward(self, audio):
         
    #     if len(audio.shape) == 2:
    #         assert audio.shape[0] == 1
    #         audio = audio.squeeze(0)
    #     elif len(audio.shape) == 3:
    #         assert audio.shape[0] == 1 and audio.shape[1] == 1
    #         audio = audio.squeeze(0).squeeze(0)

    #     return pred_vec_t(self.hubert, audio)

    def forward(self, audio, squeeze_out=True):
         
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        elif len(audio.shape) == 3:
            assert audio.shape[1] == 1
            audio = audio.squeeze(1)

        out = pred_vec_t_batch(self.hubert, audio)
        # print('forward:', out.shape, out)
        if out.shape[0] == 1 and squeeze_out:
            return out.squeeze(0)
        else:
            return out