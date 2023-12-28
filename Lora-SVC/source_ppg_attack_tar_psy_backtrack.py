
import os
import numpy as np
import torch

from scipy.io.wavfile import write
import librosa

from masker import Masker

device = 'cuda'

from content_encoder.whisperppg import WhisperPPG
from content_encoder.whisperppg_large import WhisperPPGLarge
from content_encoder.hubertvec import HubertVec
from content_encoder.s3prl import s3prl

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--start", '-start', type=int, default=0)
parser.add_argument("--end", '-end', type=int, default=-1)
parser.add_argument("--epsilon", '-epsilon', type=float, default=0.03)
parser.add_argument("--max_iter", '-max_iter', type=int, default=2000)
parser.add_argument("--lr", '-lr', type=float, default=0.0002)

parser.add_argument('--backtrack', '-backtrack', action='store_false', default=True)
parser.add_argument('--source_type', '-source_type', default=None, type=str)

parser.add_argument('-dataset', '--dataset', type=str, default='opensinger', choices=['opensinger', 'nus_cms_48'])

subparser = parser.add_subparsers(dest='system_type')

whisper_parser = subparser.add_parser("whisper")

whisper_large_parser = subparser.add_parser("whisper-large")

hubert_parser = subparser.add_parser("hubert")

whisper_large_parser = subparser.add_parser("whisper-tiny")

whisper_large_parser = subparser.add_parser("whisper-base")

whisper_large_parser = subparser.add_parser("whisper-small")

whisper_large_parser = subparser.add_parser("wav2vec2")

whisper_large_parser = subparser.add_parser("decoar2")

args = parser.parse_args()


# model
if args.system_type == 'whisper':
    content_encoder = WhisperPPG()
    print('load whisper model')
elif args.system_type == 'whisper-large':
    content_encoder = WhisperPPGLarge()
    print('load whisper Large model')
elif args.system_type == 'hubert':
    content_encoder = HubertVec()
    print('load whisper model')
elif args.system_type == 'whisper-tiny':
    content_encoder = WhisperPPG(checkpoint_name='tiny.pt')
    print('load whisper tiny model')
elif args.system_type == 'whisper-base':
    content_encoder = WhisperPPG(checkpoint_name='base.pt')
    print('load whisper base model')
elif args.system_type == 'whisper-small':
    content_encoder = WhisperPPG(checkpoint_name='small.pt')
    print('load whisper small model')
elif args.system_type == 'wav2vec2':
    content_encoder = s3prl(extractor_name='wav2vec2')
    print('load wav2vec2 model')
elif args.system_type == 'decoar2':
    content_encoder = s3prl(extractor_name='decoar2')
    print('load decoar2 model')
else:
    raise NotImplementedError
system_flag = f'-{args.system_type}' if args.system_type != 'whisper' else ''

open_singer_root_ori = './storage/OpenSinger'
backtrack_flag = '_backtrack' if args.backtrack else ''
# open_singer_root_des = f"./storage/OpenSinger-ppg_attack_tar{system_flag}-psy{backtrack_flag}-lr={str(args.lr).replace('.', '_')}"
open_singer_root_des = f"./storage/OpenSinger-ppg_attack_tar{system_flag}-psy{backtrack_flag}-lr={str(args.lr).replace('.', '_')}-test_half"
if args.source_type is not None:
    open_singer_root_des = f"./storage/OpenSinger-{args.source_type}"
if args.dataset == 'nus_cms_48':
    open_singer_root_ori = './storage/NUS-CMS-48-2'
    backtrack_flag = '_backtrack' if args.backtrack else ''
    open_singer_root_des = f"./storage/NUS-CMS-48-2-ppg_attack_tar{system_flag}-psy{backtrack_flag}-lr={str(args.lr).replace('.', '_')}"
    if args.source_type is not None:
        open_singer_root_des = f"./storage/NUS-CMS-48-2-{args.source_type}"
print(open_singer_root_ori, open_singer_root_des)

import yaml
des_file = 'select-target_speakers-source_speeches-des.yaml' if args.dataset != 'nus_cms_48' else 'select-target_speakers-source_speeches-des_NUS-CMS-48.yaml'
with open(des_file, 'r') as f:
    spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
with open(des_file, 'r') as f:
    spk_2_utt = yaml.load(f, Loader=yaml.FullLoader)
utts = []
for spk, utt in spk_2_utt.items():
    utts += utt
print(spks_keys, len(spks_keys))
print(len(utts))
utts = sorted(list(set(utts)))
print(len(utts))

paths = []
# wavs = []
for utt in utts:
    gender_flag = utt.split('_')[0]
    g = 'ManRaw' if gender_flag == 'M' else 'WomanRaw'
    name = "_".join(utt.split('_')[1:-1])
    wav_name = "_".join(utt.split('_')[1:])
    src_path = os.path.join(open_singer_root_ori, g, name, wav_name)
    os.makedirs(os.path.join(open_singer_root_des, g, name), exist_ok=True)
    des_path = os.path.join(open_singer_root_des, g, name, wav_name)
    des_path_with_backtrack = os.path.join(open_singer_root_des, g, name, wav_name.replace('.wav', '_with_backtrack.wav'))
    paths.append((src_path, des_path, des_path_with_backtrack))

    # wav, fs = librosa.load(src_path, sr=16_000)
    # assert fs == 16_000, 'sampling rate is not 16KHZ'
    # wav = torch.from_numpy(wav).float().cuda()
    # wav.requires_grad = True
    # wavs.append(wav)

my_masker = Masker(device='cuda')

back_track_path = './storage/amazing_grace.m4a'
backtrack, backtrack_sr = librosa.load(back_track_path, sr=16000)
print('backtrack:', backtrack_sr, backtrack.shape, backtrack.dtype, backtrack.max(), backtrack.min())

target_wav_path = './cxk-jntm_[cut_2sec].wav'
target_wav, target_wav_sr = librosa.load(target_wav_path, sr=16000)
print('target_wav:', target_wav_sr, target_wav.shape, target_wav.shape[0] / target_wav_sr, target_wav.dtype, target_wav.max(), target_wav.min())

N = args.max_iter
epsilon = args.epsilon
lower_ = -1
upper_ = 1
uppers = []
lowers = []

# for idx, wav in enumerate(wavs):
for idx, (src_path, _, _) in enumerate(paths):
    
    # if idx == longest_idx:
        # continue
    
    if idx not in range(args.start, args.end):
        continue

    print('>' * 10, idx, paths[idx][0], '<' * 10)

    if os.path.exists(paths[idx][1]):
        print('exists, skip')
        continue

    # try:

    wav, fs = librosa.load(src_path, sr=16_000)
    assert fs == 16_000, 'sampling rate is not 16KHZ'
    wav = torch.from_numpy(wav).float().cuda()
    wav.requires_grad = True

    with torch.no_grad():
        wav_orig = wav.clone()
        ppg_orig = content_encoder(wav_orig)
        if len(wav) < len(target_wav):
            target_wav_my = target_wav[:len(wav)]
        elif len(wav) > len(target_wav):
            target_wav_my = np.pad(target_wav, (0, len(wav)-len(target_wav)), mode='wrap')
        else:
            target_wav_my = target_wav
        target_ppg = content_encoder(torch.from_numpy(target_wav_my).float().cuda())
    
    with torch.no_grad():
        # 
        print('compute TH')
        theta_array, original_max_psd = my_masker._compute_masking_threshold(wav.detach().cpu().numpy())
        backtrack_scale = backtrack / (backtrack.max() / wav.max().item())
        theta_array_bt, original_max_psd_bt = my_masker._compute_masking_threshold(backtrack_scale[:len(wav)])
        # original_max_psd = max(original_max_psd, original_max_psd_bt)
        original_max_psd = max(original_max_psd, original_max_psd_bt) if args.backtrack else original_max_psd
        original_max_psd = torch.tensor([original_max_psd]).cuda()
        # theta_array = np.where(theta_array >= theta_array_bt, theta_array, theta_array_bt)
        theta_array = np.where(theta_array >= theta_array_bt, theta_array, theta_array_bt) if args.backtrack else theta_array
        theta_array = torch.from_numpy(theta_array).cuda().transpose(0, 1)
        print('compute TH Done')

    upper = torch.clamp(wav+epsilon, max=upper_)
    lower = torch.clamp(wav-epsilon, min=lower_)
    uppers.append(upper)
    lowers.append(lower)

    opt = torch.optim.Adam([wav], lr=args.lr)
    loss_avg = torch.zeros(2, device=device, dtype=wav.dtype)
    loss_var = torch.zeros(2, device=device, dtype=wav.dtype)
    
    for cur_iter in range(N):
        ppg = content_encoder(wav)
        # loss_1 = torch.nn.functional.cosine_similarity(ppg, ppg_orig).mean()
        loss_1 = -1. * torch.nn.functional.cosine_similarity(ppg, target_ppg).mean()
        # loss_1 = torch.nn.functional.mse_loss(ppg, target_ppg)

        delta = wav - wav_orig
        loss_th = my_masker.batch_forward_2nd_stage(delta.unsqueeze(0), theta_array.unsqueeze(0), original_max_psd.unsqueeze(0))

        all_loss = torch.zeros(2, device=device, dtype=wav.dtype)
        all_loss[0] = loss_1
        all_loss[1] = loss_th

        # scale loss
        for l_idx in range(len(all_loss)):
            loss_avg[l_idx] = loss_avg[l_idx] + (1.0 / (cur_iter+1)) * (all_loss[l_idx].item() - loss_avg[l_idx])
            loss_var[l_idx] = loss_var[l_idx] + (1.0 / (cur_iter+1)) * ((all_loss[l_idx].item() - loss_avg[l_idx]) ** 2 - loss_var[l_idx])
            if cur_iter > 0:
                if loss_var[l_idx] == 0.:
                    loss_var_eps = 1e-8
                else:
                    loss_var_eps = loss_var[l_idx]
                all_loss[l_idx] = (all_loss[l_idx] - loss_avg[l_idx]) / (loss_var_eps ** 0.5)
            else:
                all_loss[l_idx] = all_loss[l_idx] - loss_avg[l_idx]
        
        loss_tol = torch.sum(all_loss)
        
        opt.zero_grad()
        loss_tol.backward()
        opt.step()
        # wav.data =  torch.min(torch.max(wav.data, lower), upper)
        wav.data =  torch.clamp(wav.data, lower_, upper_) # no eps

        # print(cur_iter, loss_1.item(), loss_th.item(), loss_tol.item(), max(torch.abs(wav-wav_orig)).item())
        print(cur_iter, loss_1.item(), loss_th.item(), loss_tol.item())
    
    wav = (wav.detach().cpu().numpy() * (2 ** (16-1) - 1)).astype(np.int16)
    write(paths[idx][1], 16000, wav)

    backtrack_scale = (backtrack / (backtrack.max() / wav_orig.max().item()))[:len(wav)]
    backtrack_scale = (backtrack_scale * (2 ** (16-1) - 1)).astype(np.int16)
    wav_backtrack = np.stack([backtrack_scale, wav]).T
    write(paths[idx][2], 16000, wav_backtrack)

    # except (torch.cuda.OutOfMemoryError, RuntimeError):
    #     print('error, skip')
    #     continue
