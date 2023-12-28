import os

storage_dir = os.path.abspath('./storage')
os.makedirs(storage_dir, exist_ok=True)
open_singer_root_ori = os.path.join(storage_dir, 'OpenSinger')
open_singer_root = os.path.abspath(os.path.join('../', 'OpenSinger'))
assert os.path.exists(open_singer_root)
if not os.path.exists(open_singer_root_ori):
    os.system(f'ln -s {open_singer_root} {open_singer_root_ori}')
data_root = 'data_svc-all_singers-all_vocies'
os.makedirs(storage_dir + '/' + data_root, exist_ok=True)
if not os.path.exists(data_root):
    os.system('ln -s ' + storage_dir + '/' + data_root + ' ./')


class Singer():

    def __init__(self, name):
        self.name = name
        self.gender, self.idx = self.name.split('_')
        assert self.gender in ['F', 'M']
        self.gender_flag = 'ManRaw' if self.gender == 'M' else 'WomanRaw'
        self.tuned = False
        
    def tune(self, force_tune=False):
        
        data_dir = os.path.join(data_root, self.name)

        # if os.path.exists(data_dir + '/.tuned.npy') and np.load(data_dir + '/.tuned.npy') == np.array([1]) and args.force_tune:
        #     print('force tune')
        #     os.system('rm -rf ' + data_dir + '/.tuned.npy')
        # if os.path.exists(data_dir + '/.tuned.npy') and np.load(data_dir + '/.tuned.npy') == np.array([1]):
        #     self.tuned = True
        #     print('already tuned')
        #     return

        wavs_raw_dir = os.path.join(data_dir, 'waves-raw')
        print(data_dir, wavs_raw_dir)
        os.makedirs(wavs_raw_dir, exist_ok=True)
        for x in os.listdir(os.path.join(open_singer_root_ori, self.gender_flag)):
            if x.split('_')[0] != self.idx:
                continue
            for name in os.listdir(os.path.join(open_singer_root_ori, self.gender_flag, x)):
                if 'wav' not in name:
                    continue
                src_path = os.path.join(open_singer_root_ori, self.gender_flag, x, name)
                des_path = os.path.join(wavs_raw_dir, name)
                if os.path.exists(des_path):
                    continue
                os.symlink(src_path, des_path)
        
        wavs_16k_dir = os.path.join(data_dir, 'waves')
        os.makedirs(wavs_16k_dir, exist_ok=True)
        command = 'python svc_preprocess_wav.py --out_dir {} --sr 16000 --in_dir {}'.format(wavs_16k_dir, wavs_raw_dir)
        os.system(command)

        # speaker_dir = os.path.join(data_dir, 'speaker')
        # os.makedirs(speaker_dir, exist_ok=True)
        # command = 'python svc_preprocess_speaker.py {} {}'.format(wavs_16k_dir, speaker_dir)
        # os.system(command)

        # ppg_dir = os.path.join(data_dir, 'whisper')
        # os.makedirs(ppg_dir, exist_ok=True)
        # command = 'python svc_preprocess_ppg.py -w {} -p {}'.format(wavs_16k_dir, ppg_dir)
        # os.system(command)

        # command = 'python svc_preprocess_f0.py --root {}'.format(data_dir)
        # os.system(command)

        # command = 'python svc_preprocess_speaker_lora.py {}'.format(data_dir)
        # os.system(command)

        # command = 'python svc_trainer.py -c config/maxgan.yaml -n {} -root {} -ckpt_dir {} -log_dir {}'.format(self.name, data_root, ckpt_dir, log_dir)
        # print(ckpt_dir, log_dir)
        # os.system(command)

        # # command = 'python svc_inference_export.py --config config/maxgan.yaml --checkpoint_path chkpt-16k/{}/{}_0090.pt --root {}'.format(self.name, self.name, data_dir)
        # command = 'python svc_inference_export.py --config config/maxgan.yaml --checkpoint_path {}/{}/{}_0090.pt --root {}'.format(ckpt_dir, self.name, self.name, data_dir)
        # # print(command)
        # os.system(command)
        
        # self.tuned = True

        # np.save(data_dir + '/.tuned', np.array([1]))
        
singers_str = []
for i in range(28):
    singers_str.append('M_{}'.format(i))
for i in range(48):
    singers_str.append('F_{}'.format(i))

for singer_str in singers_str:
    singer = Singer(singer_str)
    singer.tune()

# select 10 voices per singer
src_root = os.path.join(storage_dir, data_root)
des_root = os.path.join(storage_dir, 'data_svc-all_singers-10_voices')
spks_keys = singers_str
for spk_key in spks_keys:
    if spk_key not in os.listdir(src_root):
        continue
    src_path = os.path.join(src_root, spk_key)
    des_path = os.path.join(des_root, spk_key)
    os.makedirs(des_path, exist_ok=True)
    for name in os.listdir(src_path):
        if name == 'waves':
            os.makedirs(os.path.join(des_path, name), exist_ok=True)
            for wav_name in sorted(os.listdir(os.path.join(src_path, name)))[:10]:
                s = os.path.join(src_path, name, wav_name)
                d = os.path.join(des_path, name, wav_name)
                if os.path.exists(d):
                    continue
                command = 'ln -s {} {}'.format(s, d)
                print(command)
                os.system(command)