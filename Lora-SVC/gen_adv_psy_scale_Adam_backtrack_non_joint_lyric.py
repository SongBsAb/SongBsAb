


import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write
import os
from speaker_encoder.SV2TTS import SV2TTS

from speaker_encoder.lora_LSTM import LoraLSTM
from speaker_encoder.iv_plda import iv_plda
from speaker_encoder.vox_trainer import SpeakerNet
from speaker_encoder.xv_plda import xv_plda
from speaker_encoder.xv_coss import xv_coss
from speaker_encoder.ecapa_tdnn import ecapa_tdnn
from speaker_encoder.xv_coss import xv_coss
from speaker_encoder.resnet import resnet
from speaker_encoder.auto_speech import auto_speech

from masker import Masker
import copy

import librosa

from pathlib import Path

from content_encoder.whisperppg import WhisperPPG
from content_encoder.whisperppg_large import WhisperPPGLarge
from content_encoder.hubertvec import HubertVec

device = 'cuda'
_device = device

if __name__ == '__main__':

    # # ## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # ## mean attack; two loss; full; change eps; cross-gender
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", "-start", type=int, default=0)
    parser.add_argument("--end", "-end", type=int, default=-1)
    parser.add_argument("--epsilon", "-epsilon", type=float, default=0.03)
    parser.add_argument("--max_iter", "-max_iter", type=int, default=1000)
    parser.add_argument("--voice_num", "-voice_num", type=int, default=10)
    
    parser.add_argument('--src_flag', '-src_root', type=str, default=None)
    parser.add_argument('--des_flag', '-des_root', type=str, default=None)
    parser.add_argument('--limit_target_spk', '-limit_target_spk', action='store_false', default=True)

    parser.add_argument('--backtrack', '-backtrack', action='store_false', default=True)

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

    vox_parser = []
    for model_name in ['RawNet3', 'ResNetSE34V2', 'VGGVox']:
        if model_name == 'RawNet3':
            epoch = 1200
        elif model_name == 'ResNetSE34V2':
            epoch = 120
        else:
            epoch = 140
        flag = 'vox_trainer-{}'.format(model_name)
        s = subparser.add_parser(flag)
        vox_parser += [s]
        s.add_argument('-checkpoint', default='{}/vox_trainer-{}/{}'.format('pre-trained-models', model_name, "model%09d.model"%epoch))

        # flag = 'vox_trainer-{}-{}-epoch'.format(model_name, epoch)
        # s = subparser.add_parser(flag)
        # s.add_argument('-checkpoint', default='{}/vox_trainer-{}/{}'.format('pre-trained-models', model_name, "model%09d.model"%epoch))

    ############### for content encoder ####################################################
    parser.add_argument("--lr", '-lr', type=float, default=0.001)

    for system_type_parser in [iv_parser, xv_parser, ecapa_parser, xv_coss_parser, resnet18_iden_parser, resnet18_veri_parser, 
                               resnet34_iden_parser, resnet34_veri_parser, autospeech_veri_parser, ge2e_parser, lora_lstm_parser] + vox_parser:
        subparser_c = system_type_parser.add_subparsers(dest='system_type_c')

        whisper_parser = subparser_c.add_parser("whisper")

        whisper_large_parser = subparser_c.add_parser("whisper-large")

        hubert_parser = subparser_c.add_parser("hubert")

    args = parser.parse_args()

    model_dir_1 = 'saved_models'
    model_dir_2 = 'pre-trained-models'
    args.dataset = 'vox2'
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
        enc_model_fpath = Path("{}/{}-{}_train-part_a/encoder_315000.bak".format(model_dir_1, args.dataset, args.model))
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

    # model
    if args.system_type_c == 'whisper':
        content_encoder = WhisperPPG()
        print('load whisper model')
    elif args.system_type_c == 'whisper-large':
        content_encoder = WhisperPPGLarge()
        print('load whisper Large model')
    elif args.system_type_c == 'hubert':
        content_encoder = HubertVec()
        print('load whisper model')
    else:
        raise NotImplementedError
    system_flag_c = f'-{args.system_type_c}' if args.system_type_c != 'whisper' else ''

    
    src_root = './storage/data_svc-all_singers-10_voices'
    backtrack_flag = '_backtrack' if args.backtrack else ''
    des_root = './storage/data_svc-adver-{}-{}-bt={}-lr={}'.format(system_flag, system_flag_c, backtrack_flag, str(args.lr).replace('.', '_'))
    if args.src_flag is not None:
        src_root = f'./storage/data_svc-{args.src_flag}'
    if args.des_flag is not None:
        des_root = f'./storage/data_svc-{args.des_flag}'
    print(system_flag, src_root, des_root)

    with torch.no_grad():
        # ## imposters
        # imposter_embs = None
        # imposter_idx_2_flag = {}
        # for x in os.listdir(src_root):
        #     y = os.path.join(src_root, x, 'lora_speaker.npy')
        #     if not os.path.exists(y):
        #         continue
        #     z = np.load(y)
        #     z = torch.from_numpy(z).unsqueeze(0)
        #     if imposter_embs is None:
        #         imposter_embs = z
        #     else:
        #         imposter_embs = torch.cat((imposter_embs, z), dim=0)
        #     imposter_idx_2_flag[imposter_embs.shape[0]-1] = x
        # imposter_embs = imposter_embs.cuda()
        ## imposters
        gender_2_imposter_embs = {'F': None, 'M': None}
        gender_2_imposter_idx_2_flag = {'F': {}, 'M': {}}
        for x in os.listdir(src_root):
            # y = os.path.join(src_root, x, 'lora_speaker.npy')
            y = os.path.join(src_root, x, f'lora_speaker{system_flag}.npy')
            if not os.path.exists(y):
                continue
            z = np.load(y)
            z = torch.from_numpy(z).unsqueeze(0)
            g = x.split('_')[0]
            if gender_2_imposter_embs[g] is None:
                gender_2_imposter_embs[g] = z
            else:
                gender_2_imposter_embs[g] = torch.cat((gender_2_imposter_embs[g], z), dim=0)
            gender_2_imposter_idx_2_flag[g][gender_2_imposter_embs[g].shape[0]-1] = x

    all_spk_keys = sorted(os.listdir(src_root))
    import yaml
    des_file = 'select-target_speakers-source_speeches-des.yaml'
    with open(des_file, 'r') as f:
        spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
        print(len(spks_keys))
    
    if args.limit_target_spk:
        spks_keys = [x for x in all_spk_keys if x in spks_keys]
    else:
        spks_keys = all_spk_keys
    print(len(spks_keys), spks_keys)
    
    my_masker = Masker(device='cuda')

    back_track_path = './storage/amazing_grace.m4a'
    backtrack, backtrack_sr = librosa.load(back_track_path, sr=16000)
    print('backtrack:', backtrack_sr, backtrack.shape, backtrack.dtype, backtrack.max(), backtrack.min())

    target_wav_path = './cxk-jntm_[cut_2sec].wav'
    target_wav, target_wav_sr = librosa.load(target_wav_path, sr=16000)
    print('target_wav:', target_wav_sr, target_wav.shape, target_wav.shape[0] / target_wav_sr, target_wav.dtype, target_wav.max(), target_wav.min())

    args.end = len(spks_keys) if args.end == -1 else args.end
    for idx_spk, flag in enumerate(spks_keys[args.start:args.end]):

        paths = []
        wavs = []

        try:
            g_dir = os.path.join(src_root, flag, 'waves')
            des_g_dir = os.path.join(des_root, flag, 'waves')
            des_g_dir_backtrack = os.path.join(des_root, flag, 'waves-with_backtrack')
            os.makedirs(des_g_dir, exist_ok=True)
            os.makedirs(des_g_dir_backtrack, exist_ok=True)
            for name in sorted(os.listdir(g_dir))[:args.voice_num]:
                if 'wav' not in name:
                    continue
                src_path = os.path.join(g_dir, name)
                des_path = os.path.join(des_g_dir, name)
                des_path_backtrack = os.path.join(des_g_dir_backtrack, name)
                paths.append((src_path, des_path, idx_spk, des_path_backtrack))
                # wav = load_waveform(src_path)
                wav, fs = torchaudio.load(src_path)
                assert fs == 16_000, 'sampling rate is not 16KHZ'
                wav = wav.squeeze(0).cuda()
                wav.requires_grad = True
                wavs.append(wav)
            
            exist = True
            for idx, wav in enumerate(wavs):
                if not os.path.exists(paths[idx][1]):
                    exist = False
            if exist:
                print('Exist, skip')
                # del wavs
                # gc.collect()
                # torch.cuda.empty_cache()
                continue
            else:
                print(f'start for {idx_spk} {flag}')
            
            with torch.no_grad():
                ori_embs = []
                ori_emb_mean = speaker_encoder.embedding(wavs[0].unsqueeze(0).unsqueeze(1))
                # ori_embs.append(ori_emb_mean)
                ori_embs.append(ori_emb_mean.clone()) # avoid influcing
                for wav in wavs[1:]:
                    pp_emb = speaker_encoder.embedding(wav.unsqueeze(0).unsqueeze(1))
                    ori_emb_mean += pp_emb
                    ori_embs.append(pp_emb)
                ori_emb_mean /= len(wavs)
                ori_emb_mean = ori_emb_mean.clone()
                ori_emb_mean = ori_emb_mean.squeeze(0)

                # imposter_sims = torch.nn.functional.cosine_similarity(ori_emb_mean.unsqueeze(0).unsqueeze(-1), imposter_embs.transpose(0, 1).unsqueeze(0), dim=1)
                # imposter_sims = imposter_sims.squeeze(0).detach().cpu().numpy()
                # imposter_idx = np.argmin(imposter_sims).flatten()[0]
                # imposter_emb = imposter_embs[imposter_idx]
                # print(flag, imposter_idx_2_flag[imposter_idx])
                my_g = flag.split('_')[0]
                opp_g = 'F' if my_g == 'M' else 'M'
                imposter_embs = gender_2_imposter_embs[opp_g].cuda()
                imposter_idx_2_flag = gender_2_imposter_idx_2_flag[opp_g]
                # imposter_sims = torch.nn.functional.cosine_similarity(ori_emb_mean.unsqueeze(0).unsqueeze(-1), imposter_embs.transpose(0, 1).unsqueeze(0), dim=1)
                imposter_sims = speaker_encoder.scoring_trials(imposter_embs, ori_emb_mean.unsqueeze(0))
                imposter_sims = imposter_sims.squeeze(0).detach().cpu().numpy()
                imposter_idx = np.argmin(imposter_sims).flatten()[0]
                imposter_emb = imposter_embs[imposter_idx]
                print(flag, imposter_idx_2_flag[imposter_idx])

            # with torch.no_grad():
            #     # 
            #     print('compute TH')
            #     all_theta_array = []
            #     all_original_max_psd = []
            #     for wav in wavs:
            #         theta_array, original_max_psd = my_masker._compute_masking_threshold(wav.detach().cpu().numpy())
            #         theta_array = torch.from_numpy(theta_array).cuda().transpose(0, 1)
            #         # print(theta_array.shape)
            #         all_theta_array.append(theta_array)
            #         all_original_max_psd.append(original_max_psd)
            #     all_original_max_psd = torch.tensor(all_original_max_psd).cuda()
            #     # print(all_original_max_psd.shape)
            #     print('compute TH Done')
            #     origin_wavs = copy.deepcopy(wavs)

            
            with torch.no_grad():
                # 
                print('compute TH')
                all_theta_array = []
                all_original_max_psd = []
                for wav in wavs:
                    theta_array, original_max_psd = my_masker._compute_masking_threshold(wav.detach().cpu().numpy())
                    backtrack_scale = backtrack / (backtrack.max() / wav.max().item())
                    theta_array_bt, original_max_psd_bt = my_masker._compute_masking_threshold(backtrack_scale[:len(wav)])
                    # original_max_psd = max(original_max_psd, original_max_psd_bt)
                    original_max_psd = max(original_max_psd, original_max_psd_bt) if args.backtrack else original_max_psd
                    theta_array = np.where(theta_array >= theta_array_bt, theta_array, theta_array_bt) if args.backtrack else theta_array
                    theta_array = torch.from_numpy(theta_array).cuda().transpose(0, 1)
                    # print(theta_array.shape)
                    all_theta_array.append(theta_array)
                    all_original_max_psd.append(original_max_psd)
                all_original_max_psd = torch.tensor(all_original_max_psd).cuda()
                # print(all_original_max_psd.shape)
                print('compute TH Done')
                origin_wavs = copy.deepcopy(wavs)

            N = args.max_iter
            epsilon = args.epsilon
            step_size = epsilon / 5
            lower_ = -1
            upper_ = 1
            uppers = []
            lowers = []

            for idx, wav in enumerate(wavs):
                upper = torch.clamp(wav+epsilon, max=upper_)
                lower = torch.clamp(wav-epsilon, min=lower_)
                uppers.append(upper)
                lowers.append(lower)
                # wav.data = wav.data + torch.tensor(np.random.uniform(-epsilon, epsilon, \
                #                     wav.shape[0]), device=wav.device, dtype=wav.dtype)
                # wav.data = wav.data + torch.tensor(np.random.uniform(-0.0001, 0.0001, \
                #                     wav.shape[0]), device=wav.device, dtype=wav.dtype)
                # wav.data =  torch.min(torch.max(wav.data, lowers[idx]), uppers[idx])
            
            for idx, wav in enumerate(wavs):

                if os.path.exists(paths[idx][1]):
                    print(f'{paths[idx][1]} Exist, skip')
                    continue

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

                loss_avg = torch.zeros(4, device=device, dtype=wavs[0].dtype)
                loss_var = torch.zeros(4, device=device, dtype=wavs[0].dtype)

                opt = torch.optim.Adam([wav], lr=args.lr)

                for cur_iter in range(N):

                    wav.grad = None
                    
                    # emb = speaker_encoder.embedding(wav)
                    # emb = emb.squeeze(0)

                    all_loss = torch.zeros(4, device=device, dtype=wavs[0].dtype)
                    # loss1 = torch.nn.functional.cosine_similarity(emb, ori_embs[idx].squeeze(0), dim=0)
                    # loss2 = torch.nn.functional.cosine_similarity(emb, imposter_emb, dim=0)
                    loss1 = speaker_encoder(wav.unsqueeze(0).unsqueeze(1), enroll_embs=ori_embs[idx]).squeeze(0)
                    loss2 = speaker_encoder(wav.unsqueeze(0).unsqueeze(1), enroll_embs=imposter_emb.unsqueeze(0)).squeeze(0)
                    all_loss[0] = loss1
                    all_loss[1] = -1. * loss2

                    # th loss
                    delta = wav - origin_wavs[idx]
                    loss_th = my_masker.batch_forward_2nd_stage(delta.unsqueeze(0), all_theta_array[idx].unsqueeze(0), all_original_max_psd[idx:idx+1])
                    all_loss[2] = loss_th

                    # content loss
                    ppg = content_encoder(wav)
                    loss_ppg = torch.nn.functional.cosine_similarity(ppg, target_ppg).mean()
                    all_loss[3] = -1. * loss_ppg

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
                    # loss_tol = all_loss[1] + all_loss[2]
                    opt.zero_grad()
                    loss_tol.backward()
                    opt.step()

                    # wav.data =  torch.min(torch.max(wav.data, lowers[idx]), uppers[idx])
                    wav.data =  torch.clamp(wav.data, lower_, upper_)
                    print('*', flag, idx, cur_iter, loss1.item(), loss2.item(), loss_th.item(), loss_ppg.item(), loss_tol.item())
                    
                
                wav = (wav.detach().cpu().numpy() * (2 ** (16-1) - 1)).astype(np.int16)
                write(paths[idx][1], 16000, wav)

                backtrack_scale = (backtrack / (backtrack.max() / origin_wavs[idx].max().item()))[:len(wav)]
                backtrack_scale = (backtrack_scale * (2 ** (16-1) - 1)).astype(np.int16)
                wav_backtrack = np.stack([backtrack_scale, wav]).T
                write(paths[idx][3], 16000, wav_backtrack)

        except torch.cuda.OutOfMemoryError:
        # except NotImplementedError:
            print('OOM')
            continue