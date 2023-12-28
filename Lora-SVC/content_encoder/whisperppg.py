
import torch.nn as nn
import torch.nn.functional as F
import os

from test_whisper_ppg import load_model, pred_ppg_infer_t, pred_ppg_infer_t_batch

class WhisperPPG(nn.Module):

    def __init__(self, checkpoint_root="whisper_pretrain", checkpoint_name="medium.pt"):
        super(WhisperPPG, self).__init__()

        self.whisper = load_model(os.path.join(checkpoint_root, checkpoint_name))

    # def forward(self, audio):
         
    #     if len(audio.shape) == 2:
    #         assert audio.shape[0] == 1
    #         audio = audio.squeeze(0)
    #     elif len(audio.shape) == 3:
    #         assert audio.shape[0] == 1 and audio.shape[1] == 1
    #         audio = audio.squeeze(0).squeeze(0) 

    #     return pred_ppg_infer_t(self.whisper, audio)

    def forward(self, audio, squeeze_out=True):
         
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        elif len(audio.shape) == 3:
            assert audio.shape[1] == 1
            audio = audio.squeeze(1)

        out = pred_ppg_infer_t_batch(self.whisper, audio)
        # print('forward:', out.shape, out)
        if out.shape[0] == 1 and squeeze_out:
            return out.squeeze(0)
        else:
            return out