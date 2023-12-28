

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

from speaker_encoder.SV2TTS import SV2TTS
from speaker_encoder.vox_trainer import SpeakerNet
from pathlib import Path

device = 'cuda'
_device = device

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
# parser.add_argument("--epsilon", type=float, default=0.02)
# parser.add_argument("--max_iter", type=int, default=50)
# parser.add_argument("--voice_num", type=int, default=10)
parser.add_argument("--attack_flag", type=str, required=True)
parser.add_argument("--attack_flag_2", type=str, required=True)

parser.add_argument('-limit_target_spk', action='store_false', default=True)
parser.add_argument('-limit_source_voice', action='store_false', default=True)
parser.add_argument('-num_source_voice', type=int, default=None)

parser.add_argument('--src_root', '-src_root', type=str, default=None)
parser.add_argument('--des_root', '-des_root', type=str, default=None)

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

ge2e_parser = subparser.add_parser("GE2E")
# for epoch in [315000]:
#     s_p = subparser.add_parser("GE2E-{}-epoch".format(epoch))

for model_name in ['RawNet3', 'ResNetSE34V2', 'VGGVox']:
    if model_name == 'RawNet3':
        epoch = 1200
    elif model_name == 'ResNetSE34V2':
        epoch = 120
    else:
        epoch = 140
    flag = 'vox_trainer-{}'.format(model_name)
    s = subparser.add_parser(flag)
    s.add_argument('-checkpoint', default='{}/vox_trainer-{}/{}'.format('pre-trained-models', model_name, "model%09d.model"%epoch))

    # flag = 'vox_trainer-{}-{}-epoch'.format(model_name, epoch)
    # s = subparser.add_parser(flag)
    # s.add_argument('-checkpoint', default='{}/vox_trainer-{}/{}'.format('pre-trained-models', model_name, "model%09d.model"%epoch))

args = parser.parse_args()

model_dir_1 = 'saved_models'
model_dir_2 = 'pre-trained-models'
args.dataset_2 = 'vox2'
args.model = 'target'

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
elif args.system_type == 'GE2E':
    enc_model_fpath = Path("{}/{}-{}_train-part_a/encoder_315000.bak".format(model_dir_1, args.dataset_2, args.model))
    print(enc_model_fpath)
    args.partial_utterance_n_frames = 160
    # encoder = SV2TTS(enc_model_fpath, device=_device, partial_utterance_n_frames=args.partial_utterance_n_frames)
    encoder = SV2TTS(enc_model_fpath, device=_device, partial_utterance_n_frames=args.partial_utterance_n_frames, backward=True)
    base_model = encoder
elif 'vox_trainer' in args.system_type:
    args.checkpoint = args.checkpoint.replace('/vox_trainer', '/vox_trainer-{}'.format(args.model))
    args.checkpoint = args.checkpoint.replace('pre-trained-models', model_dir_2)
    model_name = args.system_type.split('-')[1]
    if model_name == 'RawNet3':
        encoder_type = 'ECA'
        nOut = 256
        sinc_stride = 10
        kwargs = {'encoder_type': encoder_type, 'nOut': nOut, 'sinc_stride': sinc_stride}
    # elif model_name == 'ResNetSE34V2_AP':
    elif model_name == 'ResNetSE34V2':
        log_input = True
        encoder_type = 'ASP'
        nOut = 512
        kwargs = {'log_input': log_input, 'encoder_type': encoder_type, 'nOut': nOut} 
    elif model_name == 'VGGVox':
        log_input = True
        encoder_type = 'SAP'
        nOut = 512
        kwargs = {'log_input': log_input, 'encoder_type': encoder_type, 'nOut': nOut} 
    else:
        raise NotImplementedError
    encoder = SpeakerNet(model_name, device=_device, checkpoint=args.checkpoint, **kwargs)
    print(args.checkpoint)
    base_model = encoder
else:
    raise NotImplementedError('Unsupported System Type')
speaker_encoder = base_model


data_svc = './storage/data_svc-all_singers-10_voices'
open_singer_root_ori = './storage/OpenSinger'
new_data_svc = './storage/data_svc-{}'.format(args.attack_flag)
if args.dataset == 'nus_cms_48':
    data_svc = './storage/NUS-CMS-48'
    open_singer_root_ori = './storage/NUS-CMS-48-2'
    new_data_svc = './storage/NUS-CMS-48-{}'.format(args.attack_flag)
if args.src_root is not None:
    data_svc = args.src_root
if args.des_root is not None:
    new_data_svc = args.des_root

all_spk_keys = sorted(os.listdir(data_svc))
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
# normal_self_sim_file = f'./txt_files/{args.system_type}/{args.attack_flag}-self_sim.txt'
normal_self_sim_file = f'./txt_files/{args.system_type}/{args.attack_flag}-self_sim-all_speakers-10_voices.txt'
# normal_out_sim_file = f'./txt_files/{args.system_type}/{args.attack_flag}-out_sim.txt'
normal_out_sim_file = f'./txt_files/{args.system_type}/{args.attack_flag_2}-out_sim-all_speakers-10_voices-1000_sources.txt'
# normal_out_sim_file = f'./txt_files/{args.system_type}/{args.attack_flag}-out_sim-gen_all_test_10.txt'
# normal_out_sim_file = f'./txt_files/{args.system_type}/normal-out_sim-debug.txt'
if args.dataset == 'nus_cms_48':
    normal_self_sim_file = f'{normal_self_sim_file[:-4]}_NUS-CMS-48.txt'
    normal_out_sim_file = f'{normal_out_sim_file[:-4]}_NUS-CMS-48.txt'
# w1 = open(normal_self_sim_file, 'w')
w3 = open(normal_out_sim_file, 'w')
all_self_sims = []
all_out_sims = []
with torch.no_grad():
    for idx_spk, flag in enumerate(spks_keys):

        # if flag != 'M_4':
        # # if flag != 'F_33':
        #     continue

        # my mean emb
        wav_dir = os.path.join(data_svc, flag, 'waves')
        if not os.path.exists(wav_dir):
            continue
        print('w0:', wav_dir)
        wav_names = sorted(os.listdir(wav_dir))
        subfile_num = 0
        speaker_ave = 0
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
        speaker_ave = speaker_ave / subfile_num

        # # self sims
        # self_sims = []
        # wav_dir = os.path.join(new_data_svc, flag, 'waves')
        # if not os.path.exists(wav_dir):
        #     continue
        # print('w1:', wav_dir)
        # wav_names = sorted(os.listdir(wav_dir))
        # if len(wav_names) <= 0:
        #     continue
        # for name in wav_names:
        #     if 'wav' not in name:
        #         continue
        #     wav_path = os.path.join(wav_dir, name)
        #     wav, fs = torchaudio.load(wav_path)
        #     wav = wav.cuda()
        #     sims = speaker_encoder(wav.unsqueeze(0), enroll_embs=speaker_ave).detach().cpu().numpy().flatten().tolist()
        #     self_sims += sims
        #     w1.write(f'{flag} {name} {sims[0]}\n')
        #     print('self sims:', f'{flag} {name} {sims[0]}')
        # print('*' * 5, flag, np.mean(self_sims), np.std(self_sims), np.max(self_sims), np.min(self_sims), '*' * 5)
        # all_self_sims += self_sims
            
        # adver case out voice sims
        out_voices_sims = []
        des_root = f'./storage/model_pretrain/{flag}/inference-{args.attack_flag_2}'
        # des_root = f'./storage/model_pretrain/{flag}/inference'
        if args.dataset == 'nus_cms_48':
            des_root = f'./storage/model_pretrain_NUS-CMS-48/{flag}/inference-{args.attack_flag_2}'
        print('w3:', des_root)
        if not os.path.exists(des_root):
            continue
        if len(os.listdir(des_root)) <= 0:
            continue
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
            # print('out sims:', f'{flag} {x} {sims[0]}')
        print('*' * 5, flag, np.mean(out_voices_sims), np.std(out_voices_sims), np.max(out_voices_sims), np.min(out_voices_sims), '*' * 5)
        all_out_sims += out_voices_sims

# w1.close()
w3.close()
# print('*' * 10, 'all self sims:', np.mean(all_self_sims), np.std(all_self_sims), np.max(all_self_sims), np.min(all_self_sims), '*' * 10)
print('*' * 10, 'all out sims:', np.mean(all_out_sims), np.std(all_out_sims), np.max(all_out_sims), np.min(all_out_sims), '*' * 10)