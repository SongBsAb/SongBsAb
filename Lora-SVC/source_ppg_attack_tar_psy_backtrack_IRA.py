
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

import math

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--start", '-start', type=int, default=0)
parser.add_argument("--end", '-end', type=int, default=-1)
parser.add_argument("--epsilon", '-epsilon', type=float, default=0.03)
parser.add_argument("--max_iter", '-max_iter', type=int, default=2000)
parser.add_argument("--lr", '-lr', type=float, default=0.0002)

parser.add_argument('--backtrack', '-backtrack', action='store_false', default=True)
parser.add_argument('--source_type', '-source_type', default=None, type=str)

parser.add_argument("--my_size", '-my_size', type=int, default=32)
parser.add_argument("--batch_size", '-batch_size', type=int, default=16)

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

def sample_grids(img_size, 
                sample_grid_num=32,
                grid_scale=200,
                sample_times=32):
    grid_size = img_size // grid_scale
    # print('size info:', img_size, grid_scale, grid_size, sample_grid_num, sample_times)
    sample = []
    for _ in range(sample_times):
        grids = []
        ids = np.random.randint(0, grid_scale, size=sample_grid_num)
        # rows = ids // grid_scale # wrong
        rows = ids
        # print(rows)
        for r in rows:
            grid_range = slice(r * grid_size, (r + 1) * grid_size)
            grids.append(grid_range)
        sample.append(grids)
    return sample
def sample_for_interaction(delta,
                        img_size,
                    sample_grid_num=32,
                    grid_scale=200,
                    sample_times=32):
    samples = sample_grids(
        sample_grid_num=sample_grid_num,
        grid_scale=grid_scale,
        img_size=img_size,
        sample_times=sample_times)
    only_add_one_mask = torch.zeros_like(delta).repeat(sample_times, 1)
    for i in range(sample_times):
        grids = samples[i]
        for grid in grids:
            only_add_one_mask[i:i + 1, grid] = 1
    leave_one_mask = 1 - only_add_one_mask

    return only_add_one_mask, leave_one_mask

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
open_singer_root_des = f"./storage/OpenSinger-ppg_attack_tar{system_flag}-psy{backtrack_flag}-lr={str(args.lr).replace('.', '_')}-IRA"
if args.source_type is not None:
    open_singer_root_des = f"./storage/OpenSinger-{args.source_type}"
print(open_singer_root_ori, open_singer_root_des)

import yaml
des_file = 'select-target_speakers-source_speeches-des.yaml'
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

# backtrack, backtrack_sr = librosa.load(back_track_path, sr=16000)
# print('backtrack:', backtrack_sr, backtrack.shape, backtrack.dtype, backtrack.max(), backtrack.min())
back_track_path = './storage/amazing_grace.npy'
backtrack = np.load(back_track_path)
backtrack_sr = 16_000
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

    print('>' * 10, args.system_type, idx, paths[idx][0], '<' * 10)

    if os.path.exists(paths[idx][1]):
        print('exists, skip')
        continue
    else:
        # print('>' * 10, idx, paths[idx][0], '<' * 10)
        pass

    try:

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
        loss_avg = torch.zeros(3, device=device, dtype=wav.dtype)
        loss_var = torch.zeros(3, device=device, dtype=wav.dtype)
        
        for cur_iter in range(N):

            def scale_loss(my_loss, l_idx):
                # scale loss
                loss_avg[l_idx] = loss_avg[l_idx] + (1.0 / (cur_iter+1)) * (my_loss.item() - loss_avg[l_idx])
                loss_var[l_idx] = loss_var[l_idx] + (1.0 / (cur_iter+1)) * ((my_loss.item() - loss_avg[l_idx]) ** 2 - loss_var[l_idx])
                if cur_iter > 0:
                    if loss_var[l_idx] == 0.:
                        loss_var_eps = 1e-8
                    else:
                        loss_var_eps = loss_var[l_idx]
                    my_loss = (my_loss - loss_avg[l_idx]) / (loss_var_eps ** 0.5)
                else:
                    my_loss = my_loss - loss_avg[l_idx]
                return my_loss
            
            def scale_loss_grad(my_loss, grad, l_idx):
                # scale loss
                loss_avg[l_idx] = loss_avg[l_idx] + (1.0 / (cur_iter+1)) * (my_loss - loss_avg[l_idx])
                loss_var[l_idx] = loss_var[l_idx] + (1.0 / (cur_iter+1)) * ((my_loss - loss_avg[l_idx]) ** 2 - loss_var[l_idx])
                if cur_iter > 0:
                    if loss_var[l_idx] == 0.:
                        loss_var_eps = 1e-8
                    else:
                        loss_var_eps = loss_var[l_idx]
                    my_loss = (my_loss - loss_avg[l_idx]) / (loss_var_eps ** 0.5)
                    grad = grad / (loss_var_eps ** 0.5)
                else:
                    my_loss = my_loss - loss_avg[l_idx]
                return my_loss, grad
            
            wav.grad = None
            wav_grad = 0

            ppg = content_encoder(wav)
            loss_1 = torch.nn.functional.cosine_similarity(ppg, target_ppg).mean()

            delta = wav - wav_orig
            loss_th = my_masker.batch_forward_2nd_stage(delta.unsqueeze(0), theta_array.unsqueeze(0), original_max_psd.unsqueeze(0))

            all_loss = np.zeros(3)
            all_loss[0] = loss_1.item()
            all_loss[1] = loss_th.item()

            loss_th_s = scale_loss(loss_th, l_idx=0)
            opt.zero_grad()
            loss_th_s.backward()
            wav_grad = wav.grad.detach()

            loss_1_s = scale_loss(-1 * loss_1, l_idx=1)
            opt.zero_grad()
            loss_1_s.backward(retain_graph=True)
            wav_grad += wav.grad.detach()


            my_size = args.my_size
            with torch.no_grad():
                only_add_one_mask, leave_one_mask = \
                    sample_for_interaction(delta, wav.shape[0], 
                                            sample_grid_num=my_size,
                                            # grid_scale=256,
                                            grid_scale=200,
                                            sample_times=my_size
                                            )
            
            with torch.no_grad():
                loss_zero = torch.nn.functional.cosine_similarity(ppg_orig, target_ppg).mean()
            wav_IR_grad = 0
            loss_IR = 0
            batch_size = args.batch_size
            # assert batch_size == 1
            n_run = math.ceil(my_size/ batch_size)
            for iii in range(n_run):
                only_add_one_perturbation = delta * only_add_one_mask[iii*batch_size:(iii+1)*batch_size, :]
                leave_one_out_perturbation = delta * leave_one_mask[iii*batch_size:(iii+1)*batch_size, :]
                only_add_one_wav = only_add_one_perturbation + wav_orig.unsqueeze(0)
                leave_one_out_wav = leave_one_out_perturbation + wav_orig.unsqueeze(0)
                only_add_one_ppg = content_encoder(only_add_one_wav) # (bs, l, d)
                leave_one_out_ppg = content_encoder(leave_one_out_wav)
                only_add_one_loss = torch.nn.functional.cosine_similarity(only_add_one_ppg, target_ppg.unsqueeze(0), dim=-1).mean(dim=-1) # (bs, )
                leave_one_out_loss = torch.nn.functional.cosine_similarity(leave_one_out_ppg, target_ppg.unsqueeze(0), dim=-1).mean(dim=-1)

                # ppg = content_encoder(wav)
                # loss_1 = torch.nn.functional.cosine_similarity(ppg, target_ppg).mean()
                loss_IR_ = (loss_1 - only_add_one_loss - leave_one_out_loss + loss_zero).sum()

                opt.zero_grad()
                loss_IR_.backward(retain_graph=True)
                wav_IR_grad += wav.grad.detach()
                loss_IR += loss_IR_.item()
            wav_IR_grad /= my_size
            loss_IR /= my_size
            _, wav_IR_grad = scale_loss_grad(loss_IR, wav_IR_grad, 2)
            wav_grad += wav_IR_grad
            all_loss[2] = loss_IR

            wav.grad = wav_grad
            opt.step()
            wav.data =  torch.clamp(wav.data, lower_, upper_) # no eps

            print(args.batch_size, idx, cur_iter, all_loss)
        
        wav = (wav.detach().cpu().numpy() * (2 ** (16-1) - 1)).astype(np.int16)
        write(paths[idx][1], 16000, wav)

        backtrack_scale = (backtrack / (backtrack.max() / wav_orig.max().item()))[:len(wav)]
        backtrack_scale = (backtrack_scale * (2 ** (16-1) - 1)).astype(np.int16)
        wav_backtrack = np.stack([backtrack_scale, wav]).T
        write(paths[idx][2], 16000, wav_backtrack)

    # except (torch.cuda.OutOfMemoryError, RuntimeError):
    except torch.cuda.OutOfMemoryError:
        print('error, skip')
        continue