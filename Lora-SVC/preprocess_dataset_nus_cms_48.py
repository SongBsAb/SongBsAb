
import os

root = os.path.abspath('../nus-smc-corpus_48')
assert os.path.exists(root)
des_root = './storage/NUS-CMS-48'
for dir, _, files in os.walk(root):
    if 'sing' not in dir.split('/')[-1]:
        continue
    spk = dir.split('/')[-2]
    data_dir = os.path.join(des_root, spk)
    wavs_raw_dir = os.path.join(data_dir, 'waves-raw')
    os.makedirs(wavs_raw_dir, exist_ok=True)
    # wavs_16k_dir = os.path.join(data_dir, 'waves')
    # os.makedirs(wavs_16k_dir, exist_ok=True)
    for file in files:
        if file[-4:] != '.wav':
            continue
        src_path = os.path.join(dir, file)
        des_path = os.path.join(wavs_raw_dir, file)
        os.symlink(src_path, des_path)

data_root = './storage/NUS-CMS-48'
for name in sorted(os.listdir(data_root)):

    data_dir = os.path.join(data_root, name)
    wavs_raw_dir = os.path.join(data_dir, 'waves-raw')
    print(data_dir, wavs_raw_dir)
    
    wavs_16k_dir = os.path.join(data_dir, 'waves')
    os.makedirs(wavs_16k_dir, exist_ok=True)
    command = 'python svc_preprocess_wav.py --out_dir {} --sr 16000 --in_dir {}'.format(wavs_16k_dir, wavs_raw_dir)
    os.system(command)



import os

spks = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW', 'ZHIY']
genders = ['F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'M']
spk2gender = dict(zip(spks, genders))

root = './storage/NUS-CMS-48'
des_root = './storage/NUS-CMS-48-2'
for dir, _, files in os.walk(root):
    if 'waves' != dir.split('/')[-1]:
        continue
    # spk = dir.split('/')[-2]
    spk = dir.split('/')[-2].split('_')[1]
    gender = spk2gender[spk]
    g_dir = 'ManRaw' if gender == 'M' else 'WomanRaw'
    data_dir = os.path.join(des_root, g_dir, f'{spk}_random')
    os.makedirs(data_dir, exist_ok=True)
    for idx, file in enumerate(sorted(files)):
        if file[-4:] != '.wav':
            continue
        src_path = os.path.join(dir, file)
        des_path = os.path.join(data_dir, f'{spk}_random_{idx}.wav')
        os.symlink(src_path, des_path)

import os

spks = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW', 'ZHIY']
genders = ['F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'M']
spk2gender = dict(zip(spks, genders))

root = './storage/NUS-CMS-48'
for spk in sorted(os.listdir(root)):
    src_dir = os.path.join(root, spk)
    des_dir = os.path.join(root, f'{spk2gender[spk]}_{spk}')
    os.rename(src_dir, des_dir)
    print(src_dir, des_dir)