

import s3prl.hub as hub
import torch.nn as nn

# from s3prl.upstream.wav2vec2
# from s3prl.upstream.decoar2
class s3prl(nn.Module):

    def __init__(self, extractor_name, device='cuda'):

        super(s3prl, self).__init__()

        self.device = device

        assert extractor_name in dir(hub)
        self.encoder = getattr(hub, extractor_name)()
        self.encoder.eval().to(self.device)
    
    
    def forward(self, audio, squeeze_out=True):
         
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        elif len(audio.shape) == 3:
            assert audio.shape[1] == 1
            audio = audio.squeeze(1)
        
        x = audio
        xx = [x_ for x_ in x]
        out =  self.encoder(xx)["last_hidden_state"] # (bs, T, dim)

        # print('forward:', out.shape, out)
        if out.shape[0] == 1 and squeeze_out:
            return out.squeeze(0)
        else:
            return out
