

# ## $$$$$$ convert gt
# import os
# from pydub import AudioSegment
# import librosa
# import numpy as np
# from scipy.io.wavfile import write

# # root = './storage/data_svc-all_singers-10_voices/'
# root = './storage/NUS-CMS-48'


# back_track_path = './storage/amazing_grace.m4a'
# backtrack, backtrack_sr = librosa.load(back_track_path, sr=16000)
# print('backtrack:', backtrack_sr, backtrack.shape, backtrack.dtype, backtrack.max(), backtrack.min())

# for spk_id in os.listdir(root):

#     dir1 = os.path.join(root, spk_id, 'waves')
#     dir2 = os.path.join(root, spk_id, 'waves-with_backtrack')
#     dir3 = os.path.join(root, spk_id, 'waves-with_backtrack-mono')
#     os.makedirs(dir2, exist_ok=True)
#     os.makedirs(dir3, exist_ok=True)

#     for name in os.listdir(dir1):
#         path1 = os.path.join(dir1, name)
#         path2 = os.path.join(dir2, name)
        
#         wav, sr = librosa.load(path1, sr=16000)
#         origin_wav = wav
#         backtrack_scale = (backtrack / (backtrack.max() / origin_wav.max()))[:len(wav)]
#         backtrack_scale = (backtrack_scale * (2 ** (16-1) - 1)).astype(np.int16)
#         wav = (wav * (2 ** (16-1) - 1)).astype(np.int16)
#         wav_backtrack = np.stack([backtrack_scale, wav]).T
#         write(path2, 16000, wav_backtrack)

#         print(path1, path2)


#     for name in os.listdir(dir2):
#         path2 = os.path.join(dir2, name)
#         path3 = os.path.join(dir3, name)

#         sound = AudioSegment.from_wav(path2)
#         sound = sound.set_channels(1)
#         sound.export(path3, format="wav")

#         print(path2, path3)



# ## $$$$$$ convert attack
# import os
# from pydub import AudioSegment

# # root = './storage/data_svc-adver-full-mean-two-loss-psy_scale_backtrack-no-eps'
# # root = './storage/data_svc-adver-full-mean-two-loss-psy_scale-no-eps'
# root = './storage/NUS-CMS-48-adver-full-two-loss-psy_scale_backtrack-no-eps'

# for spk_id in os.listdir(root):

#     dir1 = os.path.join(root, spk_id, 'waves')
#     dir2 = os.path.join(root, spk_id, 'waves-with_backtrack')
#     dir3 = os.path.join(root, spk_id, 'waves-with_backtrack-mono')
#     os.makedirs(dir3, exist_ok=True)

#     for name in os.listdir(dir2):
#         path2 = os.path.join(dir2, name)
#         path3 = os.path.join(dir3, name)

#         if os.path.exists(path3):
#             continue

#         sound = AudioSegment.from_wav(path2)
#         sound = sound.set_channels(1)
#         sound.export(path3, format="wav")

#         print(path2, path3)



# $$$$$$ cal sim no write to file
# import os
# import librosa
# from pesq import pesq
# import numpy as np

# root1 = './storage/data_svc-all_singers-10_voices'
# root2 = './storage/data_svc-adver-full-mean-two-loss-psy_scale_backtrack-no-eps'
# root3 = './storage/data_svc-adver-full-mean-two-loss-psy_scale-no-eps'

# all_pesq1 = []
# all_pesq2 = []
# for spk_id in os.listdir(root1):

#     dir1 = os.path.join(root1, spk_id, 'waves-with_backtrack-mono')
#     dir2 = os.path.join(root2, spk_id, 'waves-with_backtrack-mono')
#     dir3 = os.path.join(root3, spk_id, 'waves-with_backtrack-mono')

#     for name in os.listdir(dir1):
#         path1 = os.path.join(dir1, name)
#         path2 = os.path.join(dir2, name)
#         path3 = os.path.join(dir3, name)

#         w1, sr1 = librosa.load(path1, sr=16000)
#         w2, sr2 = librosa.load(path2, sr=16000)
#         w3, sr3 = librosa.load(path3, sr=16000)

#         pesq1 = pesq(sr1, w1, w2, mode='wb')
#         pesq2 = pesq(sr1, w1, w3, mode='wb')

#         print(name, pesq1, pesq2)

#         all_pesq1.append(pesq1)
#         all_pesq2.append(pesq2)
# print(np.mean(all_pesq1), np.mean(all_pesq2))



# $$$$$$ cal sim write to file
# import os
# import librosa
# from pesq import pesq
# import numpy as np
# from metric import get_all_metric

# root1 = './storage/data_svc-all_singers-10_voices'
# root2 = './storage/data_svc-adver-full-mean-two-loss-psy_scale_backtrack-no-eps'
# root3 = './storage/data_svc-adver-full-mean-two-loss-psy_scale-no-eps'

# imper_file = 'test-imper.txt'
# w = open(imper_file, 'w')
# all_pesq1 = []
# all_pesq2 = []
# for spk_id in os.listdir(root1):
# # for spk_id in os.listdir(root1)[:1]:

#     dir1 = os.path.join(root1, spk_id, 'waves-with_backtrack-mono')
#     dir2 = os.path.join(root2, spk_id, 'waves-with_backtrack-mono')
#     dir3 = os.path.join(root3, spk_id, 'waves-with_backtrack-mono')

#     for name in os.listdir(dir1):
#         path1 = os.path.join(dir1, name)
#         path2 = os.path.join(dir2, name)
#         path3 = os.path.join(dir3, name)

#         w1, sr1 = librosa.load(path1, sr=16000)
#         w2, sr2 = librosa.load(path2, sr=16000)
#         w3, sr3 = librosa.load(path3, sr=16000)

#         pesq1_ori = get_all_metric(w1, w2, sr1)
#         pesq2_ori = get_all_metric(w1, w3, sr1)
#         pesq1 = [round(x, 3) for x in pesq1_ori]
#         pesq2 = [round(x, 3) for x in pesq2_ori]

#         line = spk_id + ' ' + name + ' ' + ' '.join([str(aaa) for aaa in pesq1_ori]) + ' ' + ' '.join([str(aaa) for aaa in pesq2_ori]) + '\n'
#         w.write(line)

#         print(spk_id, name, pesq1, pesq2)

#         all_pesq1.append(pesq1)
#         all_pesq2.append(pesq2)
# all_pesq1 = np.array(all_pesq1)
# all_pesq2 = np.array(all_pesq2)
# print(np.mean(all_pesq1, axis=0), np.mean(all_pesq2, axis=0))
# w.close()



# #$$$$$$ cal sim write to file; single attack
# import os
# import librosa
# from pesq import pesq
# import numpy as np
# from metric import get_all_metric
# from pydub import AudioSegment

# root1 = './storage/data_svc-all_singers-10_voices'
# # flag = 'adver-full-mean-two-loss-psy_scale-no-eps'
# # flag = 'adver-full-two-loss-ME_GMM+TDNN-psy_scale_backtrack-no-eps-IRA_tar_untar_32_32-lr=0.001'
# flag = 'adver-full-two-loss-psy_scale_backtrack-no-eps'
# root = f'./storage/data_svc-{flag}'

# # root1 = './storage/NUS-CMS-48'
# # flag = 'adver-full-two-loss-psy_scale_backtrack-no-eps'
# # root = f'./storage/NUS-CMS-48-{flag}'

# imper_file = f'test-imper-{flag}.txt'
# # imper_file = f'test-imper-{flag}-lr=0.0002.txt'
# # imper_file = f'test-imper-{flag}-NUS_CMS_48.txt'
# w = open(imper_file, 'w')
# all_pesq1 = []
# for spk_id in os.listdir(root):

#     # if spk_id not in ['F_MPOL', 'F_ADIZ']:
#     #     continue

#     dir1 = os.path.join(root, spk_id, 'waves-with_backtrack')
#     if not os.path.exists(dir1):
#         continue

#     for name in os.listdir(dir1):
#         if '-mono' in name:
#             continue
#         path1 = os.path.join(dir1, name)
#         path2 = os.path.join(dir1, name[:-4] + '-mono.wav')

#         sound = AudioSegment.from_wav(path1)
#         sound = sound.set_channels(1)
#         sound.export(path2, format="wav")

#         path1 = os.path.join(root1, spk_id, 'waves-with_backtrack-mono', name)

#         w1, sr1 = librosa.load(path1, sr=16000)
#         w2, sr2 = librosa.load(path2, sr=16000)

#         pesq1_ori = get_all_metric(w1, w2, sr1)
#         pesq1 = [round(x, 3) for x in pesq1_ori]

#         line = spk_id + ' ' + name + ' ' + ' '.join([str(aaa) for aaa in pesq1_ori]) + '\n'
#         w.write(line)

#         print(spk_id, name, pesq1)

#         all_pesq1.append(pesq1)

#         os.system(f'rm -rf {path2}')

# all_pesq1 = np.array(all_pesq1)
# print(np.mean(all_pesq1, axis=0))
# w.close()


# #$$$$$$ cal sim write to file; single attack; loop for transfer attack
# import os
# import librosa
# from pesq import pesq
# import numpy as np
# from metric import get_all_metric
# from pydub import AudioSegment

# root1 = './storage/data_svc-all_singers-10_voices'
# system_types = ['xv_coss', 'ecapa_tdnn', 'resnet18_iden', 
#                 'resnet34_iden', 'resnet34_veri', 'autospeech_veri', 
#                 'vox_trainer-VGGVox', 'vox_trainer-ResNetSE34V2']
# for system_type in system_types:
#     flag = f'adver-full-two-loss-{system_type}-psy_scale_backtrack-no-eps-IRA_untar_fix_row'
#     root = f'./storage/data_svc-{flag}'

#     imper_file = f'test-imper-{flag}.txt'
#     w = open(imper_file, 'w')
#     all_pesq1 = []
#     for spk_id in os.listdir(root):

#         dir1 = os.path.join(root, spk_id, 'waves-with_backtrack')
#         if not os.path.exists(dir1):
#             continue

#         for name in os.listdir(dir1):
#             if '-mono' in name:
#                 continue
#             path1 = os.path.join(dir1, name)
#             path2 = os.path.join(dir1, name[:-4] + '-mono.wav')

#             sound = AudioSegment.from_wav(path1)
#             sound = sound.set_channels(1)
#             sound.export(path2, format="wav")

#             path1 = os.path.join(root1, spk_id, 'waves-with_backtrack-mono', name)

#             w1, sr1 = librosa.load(path1, sr=16000)
#             w2, sr2 = librosa.load(path2, sr=16000)

#             pesq1_ori = get_all_metric(w1, w2, sr1)
#             pesq1 = [round(x, 3) for x in pesq1_ori]

#             line = spk_id + ' ' + name + ' ' + ' '.join([str(aaa) for aaa in pesq1_ori]) + '\n'
#             w.write(line)

#             print(spk_id, name, pesq1)

#             all_pesq1.append(pesq1)

#             os.system(f'rm -rf {path2}')

#     all_pesq1 = np.array(all_pesq1)
#     print(np.mean(all_pesq1, axis=0))
#     w.close()



# $$$$$$ cal sim write to file; no-backtrack
# import os
# import librosa
# from pesq import pesq
# import numpy as np
# from metric import get_all_metric

# root1 = './storage/data_svc-all_singers-10_voices'
# root2 = './storage/data_svc-adver-full-mean-two-loss-psy_scale_backtrack-no-eps'
# root3 = './storage/data_svc-adver-full-mean-two-loss-psy_scale-no-eps'

# imper_file = 'test-imper-voice.txt'
# w = open(imper_file, 'w')
# all_pesq1 = []
# all_pesq2 = []
# for spk_id in os.listdir(root1):
# # for spk_id in os.listdir(root1)[:1]:

#     dir1 = os.path.join(root1, spk_id, 'waves')
#     dir2 = os.path.join(root2, spk_id, 'waves')
#     dir3 = os.path.join(root3, spk_id, 'waves')

#     for name in os.listdir(dir1):
#         path1 = os.path.join(dir1, name)
#         path2 = os.path.join(dir2, name)
#         path3 = os.path.join(dir3, name)

#         w1, sr1 = librosa.load(path1, sr=16000)
#         w2, sr2 = librosa.load(path2, sr=16000)
#         w3, sr3 = librosa.load(path3, sr=16000)

#         pesq1_ori = get_all_metric(w1, w2, sr1)
#         pesq2_ori = get_all_metric(w1, w3, sr1)
#         pesq1 = [round(x, 3) for x in pesq1_ori]
#         pesq2 = [round(x, 3) for x in pesq2_ori]

#         line = spk_id + ' ' + name + ' ' + ' '.join([str(aaa) for aaa in pesq1_ori]) + ' ' + ' '.join([str(aaa) for aaa in pesq2_ori]) + '\n'
#         w.write(line)

#         print(spk_id, name, pesq1, pesq2)

#         all_pesq1.append(pesq1)
#         all_pesq2.append(pesq2)
# all_pesq1 = np.array(all_pesq1)
# all_pesq2 = np.array(all_pesq2)
# print(np.mean(all_pesq1, axis=0), np.mean(all_pesq2, axis=0))
# w.close()



# ## $$$$$$ cal sim write to file; no-backtrack; single
# import os
# import librosa
# from pesq import pesq
# import numpy as np
# from metric import get_all_metric

# root1 = './storage/data_svc-all_singers-10_voices'
# flag = 'adver-full-two-loss-psy_scale_backtrack-no-eps'
# root2 = f'./storage/data_svc-{flag}'

# imper_file = f'test-imper-voice-{flag}.txt'
# w = open(imper_file, 'w')
# all_pesq1 = []
# for spk_id in os.listdir(root1):

#     dir1 = os.path.join(root1, spk_id, 'waves')
#     dir2 = os.path.join(root2, spk_id, 'waves')

#     for name in os.listdir(dir1):
#         path1 = os.path.join(dir1, name)
#         path2 = os.path.join(dir2, name)

#         if not os.path.exists(path2):
#             continue

#         w1, sr1 = librosa.load(path1, sr=16000)
#         w2, sr2 = librosa.load(path2, sr=16000)

#         pesq1_ori = get_all_metric(w1, w2, sr1)
#         pesq1 = [round(x, 3) for x in pesq1_ori]

#         line = spk_id + ' ' + name + ' ' + ' '.join([str(aaa) for aaa in pesq1_ori]) + '\n'
#         w.write(line)

#         print(spk_id, name, pesq1)

#         all_pesq1.append(pesq1)
# all_pesq1 = np.array(all_pesq1)
# print(np.mean(all_pesq1, axis=0))
# w.close()


## $$$$$$ cal sim write to file; no-backtrack; single; loop for transfer
import os
import librosa
from pesq import pesq
import numpy as np
from metric import get_all_metric

root1 = './storage/data_svc-all_singers-10_voices'
system_types = ['xv_coss', 'ecapa_tdnn', 'resnet18_iden', 
                'resnet34_iden', 'resnet34_veri', 'autospeech_veri', 
                'vox_trainer-VGGVox', 'vox_trainer-ResNetSE34V2']
for system_type in system_types:
    flag = f'adver-full-two-loss-{system_type}-psy_scale_backtrack-no-eps-IRA_untar_fix_row'
    root2 = f'./storage/data_svc-{flag}'

    imper_file = f'test-imper-voice-{flag}.txt'
    w = open(imper_file, 'w')
    all_pesq1 = []
    for spk_id in os.listdir(root1):

        dir1 = os.path.join(root1, spk_id, 'waves')
        dir2 = os.path.join(root2, spk_id, 'waves')

        for name in os.listdir(dir1):
            path1 = os.path.join(dir1, name)
            path2 = os.path.join(dir2, name)

            if not os.path.exists(path2):
                continue

            w1, sr1 = librosa.load(path1, sr=16000)
            w2, sr2 = librosa.load(path2, sr=16000)

            pesq1_ori = get_all_metric(w1, w2, sr1)
            pesq1 = [round(x, 3) for x in pesq1_ori]

            line = spk_id + ' ' + name + ' ' + ' '.join([str(aaa) for aaa in pesq1_ori]) + '\n'
            w.write(line)

            print(spk_id, name, pesq1)

            all_pesq1.append(pesq1)
    all_pesq1 = np.array(all_pesq1)
    print(np.mean(all_pesq1, axis=0))
    w.close()



########################################################################################################

# ## $$$ convert gt
# import os
# import librosa
# import numpy as np
# from scipy.io.wavfile import write
# from pydub import AudioSegment

# # root1 = './storage/OpenSinger'
# root1 = './storage/NUS-CMS-48-2'

# import yaml
# # des_file = 'select-target_speakers-source_speeches-des.yaml'
# des_file = 'select-target_speakers-source_speeches-des_NUS-CMS-48.yaml'
# with open(des_file, 'r') as f:
#     spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
# with open(des_file, 'r') as f:
#     spk_2_utt = yaml.load(f, Loader=yaml.FullLoader)
# utts = []
# for spk, utt in spk_2_utt.items():
#     utts += utt
# print(spks_keys, len(spks_keys))
# print(len(utts))
# utts = sorted(list(set(utts)))
# print(len(utts))

# back_track_path = './storage/amazing_grace.m4a'
# backtrack, backtrack_sr = librosa.load(back_track_path, sr=16000)
# print('backtrack:', backtrack_sr, backtrack.shape, backtrack.dtype, backtrack.max(), backtrack.min())

# for g in ['WomanRaw', 'ManRaw']:
#     g_dir = os.path.join(root1, g)
#     for x in os.listdir(g_dir):
#         if len(x.split('_')) != 2:
#             continue
#         flag_2 = '{}_{}'.format('M' if g == 'ManRaw' else 'F', x.split('_')[0])
#         gender_flag = 'M' if g == 'ManRaw' else 'F'
#         for name in os.listdir(os.path.join(g_dir, x)):
#             if 'wav' not in name:
#                 continue
#             if 'backtrack' in name:
#                 continue
#             if gender_flag + '_' + name not in utts:
#                 continue

#             path1 = os.path.join(g_dir, x, name)
#             path2 = os.path.join(g_dir, x, name.replace('.wav', '_with_backtrack.wav'))

#             wav, sr = librosa.load(path1, sr=16000)
#             origin_wav = wav
#             backtrack_scale = (backtrack / (backtrack.max() / origin_wav.max()))[:len(wav)]
#             backtrack_scale = (backtrack_scale * (2 ** (16-1) - 1)).astype(np.int16)
#             wav = (wav * (2 ** (16-1) - 1)).astype(np.int16)
#             wav_backtrack = np.stack([backtrack_scale, wav]).T
#             write(path2, 16000, wav_backtrack)
#             print(path1, path2)

#             path3 = os.path.join(g_dir, x, name.replace('.wav', '_with_backtrack-mono.wav'))
#             sound = AudioSegment.from_wav(path2)
#             sound = sound.set_channels(1)
#             sound.export(path3, format="wav")
#             print(path2, path3)



# ## $$$ convert attack
# import os
# from scipy.io.wavfile import write
# from pydub import AudioSegment

# # root = './storage/OpenSinger'
# # root = './storage/OpenSinger-ppg_attack_tar-psy_backtrack-lr=0_0002'
# # root = './storage/OpenSinger-ppg_attack_tar-psy-lr=0_0002'

# root = './storage/NUS-CMS-48-2-ppg_attack_tar-psy_backtrack-lr=0_0002'

# import yaml
# # des_file = 'select-target_speakers-source_speeches-des.yaml'
# des_file = 'select-target_speakers-source_speeches-des_NUS-CMS-48.yaml'
# with open(des_file, 'r') as f:
#     spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
# with open(des_file, 'r') as f:
#     spk_2_utt = yaml.load(f, Loader=yaml.FullLoader)
# utts = []
# for spk, utt in spk_2_utt.items():
#     utts += utt
# print(spks_keys, len(spks_keys))
# print(len(utts))
# utts = sorted(list(set(utts)))
# print(len(utts))

# for g in ['WomanRaw', 'ManRaw']:
#     g_dir = os.path.join(root, g)
#     for x in os.listdir(g_dir):
#         if len(x.split('_')) != 2:
#             continue
#         flag_2 = '{}_{}'.format('M' if g == 'ManRaw' else 'F', x.split('_')[0])
#         gender_flag = 'M' if g == 'ManRaw' else 'F'
#         for name in os.listdir(os.path.join(g_dir, x)):
#             if 'backtrack' not in name:
#                 continue
#             if 'backtrack-mono' in name:
#                 continue
#             if gender_flag + '_' + '_'.join(name.split('_')[:-2]) + '.wav' not in utts:
#                 continue

#             path2 = os.path.join(g_dir, x, name)
#             path3 = os.path.join(g_dir, x, name.replace('_with_backtrack.wav', '_with_backtrack-mono.wav'))

#             sound = AudioSegment.from_wav(path2)
#             sound = sound.set_channels(1)
#             sound.export(path3, format="wav")

#             print(path2, path3)




# ## $$$ cal sim 
# import os
# import librosa
# import numpy as np
# from metric import get_all_metric

# root1 = './storage/OpenSinger'
# root2 = './storage/OpenSinger-ppg_attack_tar-psy_backtrack-lr=0_0002'
# root3 = './storage/OpenSinger-ppg_attack_tar-psy-lr=0_0002'

# import yaml
# des_file = 'select-target_speakers-source_speeches-des.yaml'
# with open(des_file, 'r') as f:
#     spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
# with open(des_file, 'r') as f:
#     spk_2_utt = yaml.load(f, Loader=yaml.FullLoader)
# utts = []
# for spk, utt in spk_2_utt.items():
#     utts += utt
# print(spks_keys, len(spks_keys))
# print(len(utts))
# utts = sorted(list(set(utts)))
# print(len(utts))

# imper_file = 'test-imper-ppg_attack.txt'
# # w = open(imper_file, 'w')
# w = open(imper_file, 'a')
# all_pesq1 = []
# all_pesq2 = []

# cnt = 0
# for g in ['WomanRaw', 'ManRaw']:
#     g_dir = os.path.join(root1, g)
#     for x in os.listdir(g_dir):
#         if len(x.split('_')) != 2:
#             continue
#         flag_2 = '{}_{}'.format('M' if g == 'ManRaw' else 'F', x.split('_')[0])
#         gender_flag = 'M' if g == 'ManRaw' else 'F'
#         for name in os.listdir(os.path.join(g_dir, x)):
#             if 'wav' not in name:
#                 continue
#             if 'backtrack-mono' not in name:
#                 continue
#             if gender_flag + '_' + '_'.join(name.split('_')[:-2]) + '.wav' not in utts:
#                 continue

#             cnt += 1
#             if cnt < 844:
#                 continue

#             path1 = os.path.join(g_dir, x, name)
#             path2 = os.path.join(root2, g, x, name)
#             path3 = os.path.join(root3, g, x, name)

#             try:
#                 w1, sr1 = librosa.load(path1, sr=16000)
#                 w2, sr2 = librosa.load(path2, sr=16000)
#                 w3, sr3 = librosa.load(path3, sr=16000)

#                 pesq1_ori = get_all_metric(w1, w2, sr1)
#                 pesq2_ori = get_all_metric(w1, w3, sr1)
#                 pesq1 = [round(x, 3) for x in pesq1_ori]
#                 pesq2 = [round(x, 3) for x in pesq2_ori]

#                 spk_id = flag_2
#                 line = spk_id + ' ' + name + ' ' + ' '.join([str(aaa) for aaa in pesq1_ori]) + ' ' + ' '.join([str(aaa) for aaa in pesq2_ori]) + '\n'
#                 w.write(line)

#                 print(spk_id, name, pesq1, pesq2)

#                 all_pesq1.append(pesq1)
#                 all_pesq2.append(pesq2)
#             except:
#                 continue

# all_pesq1 = np.array(all_pesq1)
# all_pesq2 = np.array(all_pesq2)
# print(np.mean(all_pesq1, axis=0), np.mean(all_pesq2, axis=0))
# w.close()





# ## $$$ cal sim; single 
# import os
# import librosa
# import numpy as np
# from metric import get_all_metric
# from pydub import AudioSegment

# root1 = './storage/OpenSinger'
# # flag = 'ppg_attack_tar-psy_backtrack-lr=0_0002'
# flag = 'ppg_attack_tar-hubert-psy_backtrack-lr=0_0002-IRA'
# root2 = f'./storage/OpenSinger-{flag}'

# # root1 = './storage/NUS-CMS-48-2'
# # root2 = './storage/NUS-CMS-48-2-ppg_attack_tar-psy_backtrack-lr=0_0002'

# import yaml
# des_file = 'select-target_speakers-source_speeches-des.yaml'
# # des_file = 'select-target_speakers-source_speeches-des_NUS-CMS-48.yaml'
# with open(des_file, 'r') as f:
#     spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
# with open(des_file, 'r') as f:
#     spk_2_utt = yaml.load(f, Loader=yaml.FullLoader)
# utts = []
# for spk, utt in spk_2_utt.items():
#     utts += utt
# print(spks_keys, len(spks_keys))
# print(len(utts))
# utts = sorted(list(set(utts)))
# print(len(utts))

# # imper_file = 'test-imper-ppg_attack.txt'
# imper_file = f'test-imper-{flag}.txt'
# w = open(imper_file, 'w')
# all_pesq1 = []

# cnt = 0
# for g in ['WomanRaw', 'ManRaw']:
#     g_dir = os.path.join(root1, g)
#     for x in os.listdir(g_dir):
#         if len(x.split('_')) != 2:
#             continue
#         flag_2 = '{}_{}'.format('M' if g == 'ManRaw' else 'F', x.split('_')[0])
#         gender_flag = 'M' if g == 'ManRaw' else 'F'
#         for name in os.listdir(os.path.join(g_dir, x)):
#             if 'wav' not in name:
#                 continue
#             if 'backtrack-mono' not in name:
#                 continue
#             if gender_flag + '_' + '_'.join(name.split('_')[:-2]) + '.wav' not in utts:
#                 continue

#             cnt += 1
#             # if cnt < 844:
#             #     continue

#             path1 = os.path.join(g_dir, x, name)
#             path2 = os.path.join(root2, g, x, name)

#             if not os.path.exists(path2):

#                 path2_bt = path2.replace('_with_backtrack-mono.wav', '_with_backtrack.wav')
#                 assert os.path.exists(path2_bt)
#                 sound = AudioSegment.from_wav(path2_bt)
#                 sound = sound.set_channels(1)
#                 sound.export(path2, format="wav")

#             try:
#                 w1, sr1 = librosa.load(path1, sr=16000)
#                 w2, sr2 = librosa.load(path2, sr=16000)

#                 pesq1_ori = get_all_metric(w1, w2, sr1)
#                 pesq1 = [round(x, 3) for x in pesq1_ori]

#                 spk_id = flag_2
#                 line = spk_id + ' ' + name + ' ' + ' '.join([str(aaa) for aaa in pesq1_ori]) + '\n'
#                 w.write(line)

#                 print(spk_id, name, pesq1)

#                 all_pesq1.append(pesq1)
#             except:
#                 continue

# all_pesq1 = np.array(all_pesq1)
# print(np.mean(all_pesq1, axis=0))
# w.close()




# # $$$ cal sim; no backtrack
# import os
# import librosa
# from pesq import pesq
# import numpy as np
# from metric import get_all_metric

# root1 = './storage/OpenSinger'
# root2 = './storage/OpenSinger-ppg_attack_tar-psy_backtrack-lr=0_0002'
# root3 = './storage/OpenSinger-ppg_attack_tar-psy-lr=0_0002'

# import yaml
# des_file = 'select-target_speakers-source_speeches-des.yaml'
# with open(des_file, 'r') as f:
#     spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
# with open(des_file, 'r') as f:
#     spk_2_utt = yaml.load(f, Loader=yaml.FullLoader)
# utts = []
# for spk, utt in spk_2_utt.items():
#     utts += utt
# print(spks_keys, len(spks_keys))
# print(len(utts))
# utts = sorted(list(set(utts)))
# print(len(utts))

# imper_file = 'test-imper-voice-ppg_attack.txt'
# w = open(imper_file, 'w')
# all_pesq1 = []
# all_pesq2 = []

# for g in ['WomanRaw', 'ManRaw']:
#     g_dir = os.path.join(root1, g)
#     for x in os.listdir(g_dir):
#         if len(x.split('_')) != 2:
#             continue
#         flag_2 = '{}_{}'.format('M' if g == 'ManRaw' else 'F', x.split('_')[0])
#         gender_flag = 'M' if g == 'ManRaw' else 'F'
#         for name in os.listdir(os.path.join(g_dir, x)):
#             if 'wav' not in name:
#                 continue
#             if 'backtrack' in name:
#                 continue
#             if gender_flag + '_' + name not in utts:
#                 continue

#             path1 = os.path.join(g_dir, x, name)
#             path2 = os.path.join(root2, g, x, name)
#             path3 = os.path.join(root3, g, x, name)

#             w1, sr1 = librosa.load(path1, sr=16000)
#             w2, sr2 = librosa.load(path2, sr=16000)
#             w3, sr3 = librosa.load(path3, sr=16000)

#             pesq1_ori = get_all_metric(w1, w2, sr1)
#             pesq2_ori = get_all_metric(w1, w3, sr1)
#             pesq1 = [round(x, 3) for x in pesq1_ori]
#             pesq2 = [round(x, 3) for x in pesq2_ori]

#             spk_id = flag_2
#             line = spk_id + ' ' + name + ' ' + ' '.join([str(aaa) for aaa in pesq1_ori]) + ' ' + ' '.join([str(aaa) for aaa in pesq2_ori]) + '\n'
#             w.write(line)

#             print(spk_id, name, pesq1, pesq2)

#             all_pesq1.append(pesq1)
#             all_pesq2.append(pesq2)

# all_pesq1 = np.array(all_pesq1)
# all_pesq2 = np.array(all_pesq2)
# print(np.mean(all_pesq1, axis=0), np.mean(all_pesq2, axis=0))
# w.close()


# # $$$ cal sim; no backtrack; single
# import os
# import librosa
# from pesq import pesq
# import numpy as np
# from metric import get_all_metric

# root1 = './storage/OpenSinger'
# # flag = 'ppg_attack_tar-psy_backtrack-lr=0_0002'
# flag = 'ppg_attack_tar-hubert-psy_backtrack-lr=0_0002-IRA'
# root2 = f'./storage/OpenSinger-{flag}'

# import yaml
# des_file = 'select-target_speakers-source_speeches-des.yaml'
# with open(des_file, 'r') as f:
#     spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
# with open(des_file, 'r') as f:
#     spk_2_utt = yaml.load(f, Loader=yaml.FullLoader)
# utts = []
# for spk, utt in spk_2_utt.items():
#     utts += utt
# print(spks_keys, len(spks_keys))
# print(len(utts))
# utts = sorted(list(set(utts)))
# print(len(utts))

# imper_file = f'test-imper-voice-{flag}.txt'
# w = open(imper_file, 'w')
# all_pesq1 = []

# for g in ['WomanRaw', 'ManRaw']:
#     g_dir = os.path.join(root1, g)
#     for x in os.listdir(g_dir):
#         if len(x.split('_')) != 2:
#             continue
#         flag_2 = '{}_{}'.format('M' if g == 'ManRaw' else 'F', x.split('_')[0])
#         gender_flag = 'M' if g == 'ManRaw' else 'F'
#         for name in os.listdir(os.path.join(g_dir, x)):
#             if 'wav' not in name:
#                 continue
#             if 'backtrack' in name:
#                 continue
#             if gender_flag + '_' + name not in utts:
#                 continue

#             path1 = os.path.join(g_dir, x, name)
#             path2 = os.path.join(root2, g, x, name)

#             w1, sr1 = librosa.load(path1, sr=16000)
#             w2, sr2 = librosa.load(path2, sr=16000)

#             pesq1_ori = get_all_metric(w1, w2, sr1)
#             pesq1 = [round(x, 3) for x in pesq1_ori]

#             spk_id = flag_2
#             line = spk_id + ' ' + name + ' ' + ' '.join([str(aaa) for aaa in pesq1_ori]) + '\n'
#             w.write(line)

#             print(spk_id, name, pesq1)

#             all_pesq1.append(pesq1)

# all_pesq1 = np.array(all_pesq1)
# print(np.mean(all_pesq1, axis=0))
# w.close()