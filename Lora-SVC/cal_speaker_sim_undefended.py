
import torch
import yaml
import os
import torchaudio
import numpy as np

from speaker_encoder.lora_LSTM import LoraLSTM
from speaker_encoder.iv_plda import iv_plda
from speaker_encoder.xv_plda import xv_plda
from speaker_encoder.xv_coss import xv_coss
from speaker_encoder.ecapa_tdnn import ecapa_tdnn
from speaker_encoder.xv_coss import xv_coss
from speaker_encoder.resnet import resnet
from speaker_encoder.auto_speech import auto_speech

device = 'cuda'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
parser.add_argument("--epsilon", type=float, default=0.02)
parser.add_argument("--max_iter", type=int, default=50)
parser.add_argument("--voice_num", type=int, default=10)

parser.add_argument('-limit_target_spk', action='store_false', default=True)
parser.add_argument('-limit_source_voice', action='store_false', default=True)
parser.add_argument('-num_source_voice', type=int, default=None)

# parser.add_argument('--src_root', '-src_root', type=str, default=None)
# parser.add_argument('--des_root', '-des_root', type=str, default=None)

parser.add_argument("--attack_flag", type=str, default=None)
parser.add_argument("--attack_flag_2", type=str, default=None)

parser.add_argument('-dataset', '--dataset', type=str, default='opensinger', choices=['opensinger', 'nus_cms_48'])

subparser = parser.add_subparsers(dest='system_type')

iv_parser = subparser.add_parser("iv_plda")
iv_parser.add_argument('-gmm', default='pre-trained-models/iv_plda/final_ubm.txt')
iv_parser.add_argument('-extractor', default='pre-trained-models/iv_plda/final_ie.txt')
iv_parser.add_argument('-plda', default='pre-trained-models/iv_plda/plda.txt')
iv_parser.add_argument('-mean', default='pre-trained-models/iv_plda/mean.vec')
iv_parser.add_argument('-transform', default='pre-trained-models/iv_plda/transform.txt')
iv_parser.add_argument('-model_file', default='model_file/iv_plda/speaker_model_iv_plda')

xv_parser = subparser.add_parser("xv_plda")
xv_parser.add_argument('-extractor', default='pre-trained-models/xv_plda/xvecTDNN_origin.ckpt')
xv_parser.add_argument('-plda', default='pre-trained-models/xv_plda/plda.txt')
xv_parser.add_argument('-mean', default='pre-trained-models/xv_plda/mean.vec')
xv_parser.add_argument('-transform', default='pre-trained-models/xv_plda/transform.txt')
xv_parser.add_argument('-model_file', default='model_file/xv_plda/speaker_model_xv_plda')

ecapa_parser = subparser.add_parser("ecapa_tdnn")
ecapa_parser.add_argument('-extractor', 
            default='pre-trained-models/ecapa_tdnn/embedding_model.ckpt')
ecapa_parser.add_argument('-mean', default='pre-trained-models/ecapa_tdnn/ecapa-tdnn-emb-mean.pickle')
ecapa_parser.add_argument('-model_file', default='model_file/ecapa_tdnn/speaker_model_ecapa_tdnn')

xv_coss_parser = subparser.add_parser("xv_coss")
xv_coss_parser.add_argument('-extractor', 
            default='pre-trained-models/xv_coss/embedding_model.ckpt')
xv_coss_parser.add_argument('-mean', default='pre-trained-models/xv_coss/xv-coss-emb-mean.pickle')
xv_coss_parser.add_argument('-model_file', default='model_file/xv_coss/speaker_model_xv_coss')

resnet18_iden_parser = subparser.add_parser("resnet18_iden")
resnet18_iden_parser.add_argument('-extractor',
            default='pre-trained-models/auto_speech/autoSpeech-resnet18-iden.pth')
resnet18_iden_parser.add_argument('-mean', default='pre-trained-models/auto_speech/resnet18_iden-emb-mean.pickle')
resnet18_iden_parser.add_argument('-feat_mean',
            default='pre-trained-models/auto_speech/resNet18-feat-mean')
resnet18_iden_parser.add_argument('-feat_std',
            default='pre-trained-models/auto_speech/resNet18-feat-std')
resnet18_iden_parser.add_argument('-model_file', default='model_file/resnet18_iden/speaker_model_resnet18_iden')

resnet18_veri_parser = subparser.add_parser("resnet18_veri")
resnet18_veri_parser.add_argument('-extractor',
            default='pre-trained-models/auto_speech/autoSpeech-resnet18-veri.pth')
resnet18_veri_parser.add_argument('-mean', default='pre-trained-models/auto_speech/resnet18_veri-emb-mean.pickle')
resnet18_veri_parser.add_argument('-feat_mean',
            default='pre-trained-models/auto_speech/resNet18-feat-mean')
resnet18_veri_parser.add_argument('-feat_std',
            default='pre-trained-models/auto_speech/resNet18-feat-std')
resnet18_veri_parser.add_argument('-model_file', default='model_file/resnet18_veri/speaker_model_resnet18_veri')

resnet34_iden_parser = subparser.add_parser("resnet34_iden")
resnet34_iden_parser.add_argument('-extractor',
            default='pre-trained-models/auto_speech/autoSpeech-resnet34-iden.pth')
resnet34_iden_parser.add_argument('-mean', default='pre-trained-models/auto_speech/resnet34_iden-emb-mean.pickle')
resnet34_iden_parser.add_argument('-feat_mean',
            default='pre-trained-models/auto_speech/resNet18-feat-mean') # common for 18 and 34
resnet34_iden_parser.add_argument('-feat_std',
            default='pre-trained-models/auto_speech/resNet18-feat-std') # common for 18 and 34
resnet34_iden_parser.add_argument('-model_file', default='model_file/resnet34_iden/speaker_model_resnet34_iden')

resnet34_veri_parser = subparser.add_parser("resnet34_veri")
resnet34_veri_parser.add_argument('-extractor',
            default='pre-trained-models/auto_speech/autoSpeech-resnet34-veri.pth')
resnet34_veri_parser.add_argument('-mean', default='pre-trained-models/auto_speech/resnet34_veri-emb-mean.pickle')
resnet34_veri_parser.add_argument('-feat_mean',
            default='pre-trained-models/auto_speech/resNet18-feat-mean') # common for 18 and 34
resnet34_veri_parser.add_argument('-feat_std',
            default='pre-trained-models/auto_speech/resNet18-feat-std') # common for 18 and 34
resnet34_veri_parser.add_argument('-model_file', default='model_file/resnet34_veri/speaker_model_resnet34_veri')

autospeech_veri_parser = subparser.add_parser("autospeech_veri")
autospeech_veri_parser.add_argument('-extractor',
            # default='pre-trained-models/auto_speech/autoSpeech-scratch-veri.pth',
            default='pre-trained-models/auto_speech/autospeech_sctrach_veri_ckpt'
            )
autospeech_veri_parser.add_argument('-mean', default='pre-trained-models/auto_speech/autospeech_veri-emb-mean.pickle')
autospeech_veri_parser.add_argument('-feat_mean',
            default='pre-trained-models/auto_speech/resNet18-feat-mean') # common for 18 and 34
autospeech_veri_parser.add_argument('-feat_std',
            default='pre-trained-models/auto_speech/resNet18-feat-std') # common for 18 and 34
autospeech_veri_parser.add_argument('-model_file', default='model_file/autospeech_veri/speaker_model_autospeech_veri')

lora_lstm_parser = subparser.add_parser("lora_LSTM")

args = parser.parse_args()

system_flag = f'-{args.system_type}' if args.system_type != 'lora_LSTM' else ''
args.threshold = -np.infty
if args.system_type == 'lora_LSTM':
    base_model = LoraLSTM()
elif args.system_type == 'iv_plda':
    base_model = iv_plda(args.gmm, args.extractor, args.plda, args.mean, args.transform, device=device, model_file=args.model_file, threshold=args.threshold)
elif args.system_type == 'xv_plda':
    base_model = xv_plda(args.extractor, args.plda, args.mean, args.transform, device=device, model_file=args.model_file, threshold=args.threshold)
elif args.system_type == 'ecapa_tdnn':
    base_model = ecapa_tdnn(args.extractor, mean_file=args.mean, device=device, model_file=args.model_file, threshold=args.threshold)
elif args.system_type == 'xv_coss':
    base_model = xv_coss(args.extractor, mean_file=args.mean, device=device, model_file=args.model_file, threshold=args.threshold)
elif 'resnet' in args.system_type:
    MODEL_NAME = args.system_type.split('_')[0]
    base_model = resnet(MODEL_NAME, args.extractor, mean_file=args.mean, 
                        feat_mean_file=args.feat_mean, feat_std_file=args.feat_std, device=device, model_file=args.model_file, threshold=args.threshold)
elif 'autospeech' in args.system_type:
    base_model = auto_speech(args.extractor, mean_file=args.mean, 
                        feat_mean_file=args.feat_mean, feat_std_file=args.feat_std, device=device, model_file=args.model_file, threshold=args.threshold)
else:
    raise NotImplementedError('Unsupported System Type')
speaker_encoder = base_model


# data_svc = './storage/data_svc' if args.src_root is None else args.src_root
data_svc = './storage/data_svc' if args.attack_flag is None else f'./storage/data_svc-{args.attack_flag}'
open_singer_root_ori = './storage/OpenSinger'
if args.dataset == 'nus_cms_48':
    data_svc = './storage/NUS-CMS-48' if args.attack_flag is None else f'./storage/NUS-CMS-48-{args.attack_flag}'
    open_singer_root_ori = './storage/NUS-CMS-48-2'
print(data_svc)

all_spk_keys = os.listdir(data_svc)
des_file = 'select-target_speakers-source_speeches-des.yaml' if args.dataset != 'nus_cms_48' else 'select-target_speakers-source_speeches-des_NUS-CMS-48.yaml'
with open(des_file, 'r') as f:
    spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
    print(len(spks_keys))
with open(des_file, 'r') as f:
    spk_2_utt = yaml.load(f, Loader=yaml.FullLoader)
if args.limit_target_spk:
    spks_keys = [x for x in all_spk_keys if x in spks_keys]
else:
    spks_keys = all_spk_keys
print(spks_keys, len(spks_keys))

os.makedirs(f'./txt_files/{args.system_type}', exist_ok=True)
normal_self_sim_file = f'./txt_files/{args.system_type}/normal-self_sim.txt'
source_sim_file = f'./txt_files/{args.system_type}/source_sim.txt'
normal_out_sim_file = f'./txt_files/{args.system_type}/normal-out_sim.txt'
# normal_self_sim_file = f'./txt_files/{args.system_type}/normal-self_sim-all_speakers-10_voices.txt'
# source_sim_file = f'./txt_files/{args.system_type}/source_sim-all_speakers-10_voices-1000_sources.txt'
# normal_out_sim_file = f'./txt_files/{args.system_type}/normal-out_sim-all_speakers-10_voices-1000_sources.txt'

if args.dataset == 'nus_cms_48':
    normal_self_sim_file = f'{normal_self_sim_file[:-4]}_NUS-CMS-48.txt'
    source_sim_file = f'{source_sim_file[:-4]}_NUS-CMS-48.txt'
    normal_out_sim_file = f'{normal_out_sim_file[:-4]}_NUS-CMS-48.txt'
# w1 = open(normal_self_sim_file, 'w')
w2 = open(source_sim_file, 'w')
w3 = open(normal_out_sim_file, 'w')
all_self_sims = []
all_source_sims = []
all_out_sims = []
with torch.no_grad():
    for idx_spk, flag in enumerate(spks_keys):
        # my mean emb
        wav_dir = os.path.join(data_svc, flag, 'waves')
        wav_names = sorted(os.listdir(wav_dir))
        subfile_num = 0
        speaker_ave = 0
        all_my_embs = []
        all_my_wavs = []
        # for name in wav_names:
        for name in wav_names[:10]:
            if 'wav' not in name:
                continue
            wav_path = os.path.join(wav_dir, name)
            wav, fs = torchaudio.load(wav_path)
            wav = wav.cuda()
            source_embed = speaker_encoder.embedding(wav.unsqueeze(0))
            speaker_ave = speaker_ave + source_embed
            subfile_num = subfile_num + 1
            all_my_embs.append(source_embed)
            all_my_wavs.append((name, wav))
        speaker_ave = speaker_ave / subfile_num
        all_my_embs = torch.cat(all_my_embs, dim=0)

        # # self sims
        # self_sims = []
        # for name, wav in all_my_wavs:
        #     sims = speaker_encoder(wav.unsqueeze(0), enroll_embs=speaker_ave).detach().cpu().numpy().flatten().tolist()
        #     self_sims += sims
        #     w1.write(f'{flag} {name} {sims[0]}\n')
        # all_self_sims += self_sims
        # print('*' * 5, flag, np.mean(self_sims), np.std(self_sims), np.max(self_sims), np.min(self_sims), '*' * 5)

        pre_dir = 'model_pretrain' if args.dataset != 'nus_cms_48' else 'model_pretrain_NUS-CMS-48'
        des_root = f'./storage/{pre_dir}/{flag}/inference' if args.attack_flag_2 is None \
            else f'./storage/{pre_dir}/{flag}/inference-{args.attack_flag_2}'
        # source voice sim
        source_voices_sims = []
        for g in ['WomanRaw', 'ManRaw']:
            g_dir = os.path.join(open_singer_root_ori, g)
            for x in os.listdir(g_dir):
                if len(x.split('_')) != 2:
                    continue
                flag_2 = '{}_{}'.format('M' if g == 'ManRaw' else 'F', x.split('_')[0])
                gender_flag = 'M' if g == 'ManRaw' else 'F'
                if flag_2 == flag:
                    continue
                cnt = 0
                for name in os.listdir(os.path.join(g_dir, x)):
                    if 'wav' not in name:
                        continue
                    # if gender_flag + '_' + name not in spk_2_utt[flag]:
                    if args.limit_source_voice and gender_flag + '_' + name not in spk_2_utt[flag]:
                        continue
                    if args.num_source_voice is not None and gender_flag + '_' + name not in os.listdir(des_root):
                        continue
                    src_path = os.path.join(g_dir, x, name)
                    src_wav, fs = torchaudio.load(src_path)
                    src_wav = src_wav.cuda()
                    sims = speaker_encoder(src_wav.unsqueeze(0), enroll_embs=speaker_ave).detach().cpu().numpy().flatten().tolist()
                    source_voices_sims += sims
                    # w2.write(f'{flag} {name} {sims[0]}\n')
                    w2.write(f'{flag} {gender_flag + "_" + name} {sims[0]}\n')
        print('*' * 5, flag, np.mean(source_voices_sims), np.std(source_voices_sims), np.max(source_voices_sims), np.min(source_voices_sims), '*' * 5)
        all_source_sims += source_voices_sims
            
        # normal case out voice sims
        out_voices_sims = []
        # des_root = f'./storage/model_pretrain/{flag}/inference'
        pre_dir = 'model_pretrain' if args.dataset != 'nus_cms_48' else 'model_pretrain_NUS-CMS-48'
        des_root = f'./storage/{pre_dir}/{flag}/inference' if args.attack_flag_2 is None \
            else f'./storage/{pre_dir}/{flag}/inference-{args.attack_flag_2}'
        for x in os.listdir(des_root):
            if 'wav' not in x:
                continue
            if 'pitch.wav' in x:
                continue
            # if x not in spk_2_utt[flag]:
            if args.limit_source_voice and x not in spk_2_utt[flag]:
                continue
            path = os.path.join(des_root, x)
            wav, fs = torchaudio.load(path)
            wav = wav.cuda()
            sims = speaker_encoder(wav.unsqueeze(0), enroll_embs=speaker_ave).detach().cpu().numpy().flatten().tolist()
            out_voices_sims += sims
            w3.write(f'{flag} {x} {sims[0]}\n')
        print('*' * 5, flag, np.mean(out_voices_sims), np.std(out_voices_sims), np.max(out_voices_sims), np.min(out_voices_sims), '*' * 5)
        all_out_sims += out_voices_sims

# w1.close()
w2.close()
w3.close()
# print('*' * 10, 'all self sims:', np.mean(all_self_sims), np.std(all_self_sims), np.max(all_self_sims), np.min(all_self_sims), '*' * 10)
print('*' * 10, 'all source sims:', np.mean(all_source_sims), np.std(all_source_sims), np.max(all_source_sims), np.min(all_source_sims), '*' * 10)
print('*' * 10, 'all out sims:', np.mean(all_out_sims), np.std(all_out_sims), np.max(all_out_sims), np.min(all_out_sims), '*' * 10)