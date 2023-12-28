import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch
import librosa

from hubert import hubert_model


def load_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x


def load_model(path, device):
    model = hubert_model.hubert_soft(path)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    return model


def pred_vec(model, wavPath, vecPath, device):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    vec_a = []
    idx_s = 0
    while (idx_s + 20 * 16000 < audln):
        feats = audio[idx_s:idx_s + 20 * 16000]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            vec_a.extend(vec)
        idx_s = idx_s + 20 * 16000
    if (idx_s < audln):
        feats = audio[idx_s:audln]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            # print(vec.shape)   # [length, dim=256] hop=320
            vec_a.extend(vec)
    np.save(vecPath, vec_a, allow_pickle=False)


def pred_vec_t(model, audio, device='cuda'):
    audln = audio.shape[0]
    vec_a = []
    idx_s = 0
    while (idx_s + 20 * 16000 < audln):
        feats = audio[idx_s:idx_s + 20 * 16000]
        feats = feats.to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        vec = model.units(feats).squeeze() # (length, dim)
        vec_a.extend(vec)
        idx_s = idx_s + 20 * 16000
    if (idx_s < audln):
        feats = audio[idx_s:audln]
        feats = feats.to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        vec = model.units(feats).squeeze()
        # print(vec.shape)   # [length, dim=256] hop=320
        vec_a.extend(vec)
    # return torch.concat(vec_a)
    return torch.stack(vec_a) # （all_length, dim）

def pred_vec_t_batch(model, audio, device='cuda'):
    audln = audio.shape[1]
    vec_a = []
    idx_s = 0
    while (idx_s + 20 * 16000 < audln):
        feats = audio[:, idx_s:idx_s + 20 * 16000]
        feats = feats.to(device)
        feats = feats[:, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        vec = model.units(feats) # (bs, length, dim)
        # print('1:', vec.shape)   # [length, dim=256] hop=320
        vec_a.append(vec)
        idx_s = idx_s + 20 * 16000
    if (idx_s < audln):
        feats = audio[:, idx_s:audln]
        feats = feats.to(device)
        feats = feats[:, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        vec = model.units(feats)
        # print('2:', vec.shape)   # [length, dim=256] hop=320
        vec_a.append(vec)
    out = torch.cat(vec_a, 1)
    # print('3:', out.shape)
    return out # (bs, T, dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-v", "--vec", help="vec", dest="vec", required=True)
    args = parser.parse_args()
    print(args.wav)
    print(args.vec)

    wavPath = args.wav
    vecPath = args.vec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hubert = load_model(os.path.join(
        "hubert_pretrain", "hubert-soft-0d54a1f4.pt"), device)
    pred_vec(hubert, wavPath, vecPath, device)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.description = 'please enter embed parameter ...'
#     parser.add_argument("-w", "--wav", help="wav", dest="wav")
#     args = parser.parse_args()
#     print(args.wav)
#     wavPath = args.wav

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     hubert = load_model(os.path.join(
#         "hubert_pretrain", "hubert-soft-0d54a1f4.pt"), device)

#     import torchaudio
#     audio = torchaudio.load(wavPath)[0].squeeze(0).cuda()
#     audio.requires_grad = True
#     print(audio, audio.shape)
    
#     ppg = pred_vec_t(hubert, audio)
#     print(ppg, ppg.shape)
#     ppg.backward(torch.ones_like(ppg)) 
#     print(audio.grad, audio.grad.shape)