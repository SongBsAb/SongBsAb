
import torch.nn as nn
import torch.nn.functional as F
import os

from whisper_so_vits_svc_5.inference import pred_ppg_t, load_model, pred_ppg_t_batch

class WhisperPPGLarge(nn.Module):

    def __init__(self, checkpoint_root="whisper_pretrain", checkpoint_name="large-v2.pt"):
        super(WhisperPPGLarge, self).__init__()

        self.whisper = load_model(os.path.join(checkpoint_root, checkpoint_name), device='cuda')

    # def forward(self, audio):
         
    #     if len(audio.shape) == 2:
    #         assert audio.shape[0] == 1
    #         audio = audio.squeeze(0)
    #     elif len(audio.shape) == 3:
    #         assert audio.shape[0] == 1 and audio.shape[1] == 1
    #         audio = audio.squeeze(0).squeeze(0) 

    #     return pred_ppg_t(self.whisper, audio)
    
    def forward(self, audio, squeeze_out=True):
         
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        elif len(audio.shape) == 3:
            assert audio.shape[1] == 1
            audio = audio.squeeze(1)

        out = pred_ppg_t_batch(self.whisper, audio)
        # print('forward:', out.shape, out)
        if out.shape[0] == 1 and squeeze_out:
            return out.squeeze(0)
        else:
            return out