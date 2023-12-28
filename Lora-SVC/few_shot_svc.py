
import os
import numpy as np

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-start', '--start', type=int, default=0)
    parser.add_argument('-end', '--end', type=int, default=-1)
    parser.add_argument('-w_start', '--w_start', type=int, default=0)
    # parser.add_argument('-w_end', '--w_end', type=int, default=100)
    parser.add_argument('-w_end', '--w_end', type=int, default=-1)

    parser.add_argument('-type', '--type', type=str, default='')
    parser.add_argument('-source_type', '--source_type', type=str, default='')
    parser.add_argument('-infer_dir', '--infer_dir', type=str, default=None)

    parser.add_argument('-force_pre', '--force_pre', action='store_true', default=False)
    parser.add_argument('-force_gen', '--force_gen', action='store_true', default=False)

    parser.add_argument('-limit_target_spk', '--limit_target_spk', action='store_false', default=True)
    parser.add_argument('-limit_source_voice', '--limit_source_voice', action='store_false', default=True)
    parser.add_argument('-num_source_voice', '--num_source_voice', type=int, default=None)

    parser.add_argument('-infer_no_pit', '--infer_no_pit', default=False, action='store_true')
    parser.add_argument('-singer_pitch_ave_factor', '--singer_pitch_ave_factor', type=float, default=0., 
                        help="Path of pitch statics.")
    parser.add_argument("--crepe", action='store_true', default=False)

    parser.add_argument('-dataset', '--dataset', type=str, default='opensinger', choices=['opensinger', 'nus_cms_48'])

    parser.add_argument("--num", type=int, default=None)

    args = parser.parse_args()

    np.random.seed(666)

    if args.source_type == '':
        source_type = ''
    else:
        source_type = f'-{args.source_type}'
    open_singer_root_ori = f'./storage/OpenSinger{source_type}'
    if args.dataset == 'nus_cms_48':
        open_singer_root_ori = f'./storage/NUS-CMS-48-2{source_type}'
    print(source_type, open_singer_root_ori)

    import os
    pre_dir = './storage/model_pretrain'
    os.makedirs(pre_dir, exist_ok=True)
    command = 'python svc_inference_export.py --config config/maxgan.yaml --checkpoint_path {} --root {}'.format(pre_dir + '/maxgan_pretrain_16K_5L.pth', 
                                                                                                                 pre_dir)
    assert os.path.exists(pre_dir + '/maxgan_pretrain_16K_5L.pth')
    print(command)
    # os.system(command)

    if args.type == '':
        # attack_type = ''
        attack_type = ''
    else:
        attack_type = f'-{args.type}'
    
    if source_type == '' and attack_type == '':
        my_flag = ''
    elif source_type == '':
        my_flag = f'{attack_type}'
    elif attack_type == '':
        my_flag = f'{source_type}'
    else:
        my_flag = f'{source_type}{attack_type}'
    print(attack_type, my_flag)
    data_root = f'data_svc{attack_type}'
    # infer_dir = f'inference{attack_type}'
    infer_dir = f'inference{my_flag}'
    if args.dataset == 'nus_cms_48':
        data_root = f'NUS-CMS-48{attack_type}'
        # infer_dir = f'inference-NUS-CMS-48{my_flag}'
    if args.infer_no_pit:
        infer_dir = f'{infer_dir}-no_pitch_shift'
    if args.singer_pitch_ave_factor != 0.:
        infer_dir = f'{infer_dir}-singer_pitch_ave_factor={args.singer_pitch_ave_factor}'
    if args.crepe:
        infer_dir = f'{infer_dir}-crepe'
    if args.num is not None:
        infer_dir = f'{infer_dir}-n={args.num}'
    if args.infer_dir is not None:
        infer_dir = args.infer_dir
    print(data_root, infer_dir)

    storage_dir = './storage'
    assert os.path.exists(storage_dir + '/' + data_root)
    if not os.path.exists(data_root):
        os.system('ln -s ' + storage_dir + '/' + data_root + ' ./')

    singers_str = []
    for i in range(28):
        singers_str.append('M_{}'.format(i))
    for i in range(48):
        singers_str.append('F_{}'.format(i))
    if args.dataset == 'nus_cms_48':
        golfer_spks = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW', 'ZHIY']
        golfer_genders = ['F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'M']
        singers_str = []
        for i in [xxx for xxx, yyy in zip(golfer_spks, golfer_genders) if yyy == 'M']:
            singers_str.append('M_{}'.format(i))
        for i in [xxx for xxx, yyy in zip(golfer_spks, golfer_genders) if yyy == 'F']:
            singers_str.append('F_{}'.format(i))
    import yaml
    des_file = 'select-target_speakers-source_speeches-des.yaml'
    if args.dataset == 'nus_cms_48':
        des_file = 'select-target_speakers-source_speeches-des_NUS-CMS-48.yaml'
    with open(des_file, 'r') as f:
        spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
    with open(des_file, 'r') as f:
        spk_2_utt = yaml.load(f, Loader=yaml.FullLoader)
    if args.limit_target_spk:
        singers_str = [x for x in singers_str if x in spks_keys]
    print(singers_str)
    start = min(max(args.start, 0), len(singers_str))
    end =  len(singers_str) if args.end == -1 else args.end
    end = min(max(end, 0), len(singers_str))
    print(start, end)
    args.start = start
    args.end = end
    flags = singers_str[args.start:args.end]

    for flag_idx, flag in enumerate(flags):
        print(flag_idx, flag, len(spk_2_utt[flag]))

    for flag_idx, flag in enumerate(flags):

        data_dir = os.path.join(data_root, flag)
        speaker_file = os.path.join(data_dir, 'lora_speaker.npy')
        speaker_file_2 = os.path.join(data_dir, 'lora_speaker.npy') if args.num is None else os.path.join(data_dir, f'lora_speaker_n={args.num}.npy')
        picth_file = os.path.join(data_dir, 'lora_pitch_statics.npy' if not args.crepe else 'lora_pitch_statics_crepe.npy')
        picth_file_2 = os.path.join(data_dir, 'lora_pitch_statics.npy' if not args.crepe else 'lora_pitch_statics_crepe.npy') if args.num is None else os.path.join(data_dir, f'lora_pitch_statics_n={args.num}.npy' if not args.crepe else f'lora_pitch_statics_crepe_n={args.num}.npy')
        condition = (not os.path.exists(speaker_file) or not os.path.exists(picth_file)) if not args.force_pre else True
        if condition:
            # generate speaker embeddings and pitch first
            wavs_16k_dir = os.path.join(data_dir, 'waves')
            speaker_dir = os.path.join(data_dir, 'speaker')
            os.makedirs(speaker_dir, exist_ok=True)
            command = 'python svc_preprocess_speaker.py "{}" "{}"'.format(wavs_16k_dir, speaker_dir)
            os.system(command)

            command = 'python svc_preprocess_f0.py --root "{}"'.format(data_dir)
            if args.crepe:
                command += ' --crepe'
            os.system(command)

            command = 'python svc_preprocess_speaker_lora.py "{}"'.format(data_dir)
            # command = 'python svc_preprocess_speaker_lora.py "{}"'.format(data_dir) if args.num is None else 'python svc_preprocess_speaker_lora.py "{}" --num {}'.format(data_dir, args.num)
            if args.crepe:
                command += ' --crepe'
            os.system(command)
        
        condition = (not os.path.exists(speaker_file_2) or not os.path.exists(picth_file_2)) if not args.force_pre else True
        if condition:
            # generate speaker embeddings and pitch first
            wavs_16k_dir = os.path.join(data_dir, 'waves')
            speaker_dir = os.path.join(data_dir, 'speaker')
            os.makedirs(speaker_dir, exist_ok=True)
            command = 'python svc_preprocess_speaker.py "{}" "{}"'.format(wavs_16k_dir, speaker_dir)
            # os.system(command)

            command = 'python svc_preprocess_f0.py --root "{}"'.format(data_dir)
            if args.crepe:
                command += ' --crepe'
            # os.system(command)

            # command = 'python svc_preprocess_speaker_lora.py "{}"'.format(data_dir)
            command = 'python svc_preprocess_speaker_lora.py "{}"'.format(data_dir) if args.num is None else 'python svc_preprocess_speaker_lora.py "{}" --num {}'.format(data_dir, args.num)
            if args.crepe:
                command += ' --crepe'
            os.system(command)
        


        paths = []
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
                    if 'backtrack' in name:
                        continue
                    # if gender_flag + '_' + name not in spk_2_utt[flag]:
                    if args.limit_source_voice and gender_flag + '_' + name not in spk_2_utt[flag]:
                        continue
                    src_path = os.path.join(g_dir, x, name)
                    des_dir = os.path.join(pre_dir, flag, infer_dir)
                    if args.dataset == 'nus_cms_48':
                        des_dir = os.path.join(f'{pre_dir}_NUS-CMS-48', flag, infer_dir)
                    os.makedirs(des_dir, exist_ok=True)
                    des_path = os.path.join(des_dir, gender_flag + '_' + name)
                    des_path_pitch = os.path.join(des_dir, gender_flag + '_' + name[:-4] + '_pitch.wav')
                    ppg_path = os.path.join(des_dir, gender_flag + '_' + name[:-4] + '.ppg.npy')
                    paths.append((src_path, des_path, des_path_pitch, ppg_path))
                    cnt += 1
        print(len(paths))
        if args.num_source_voice is not None and args.num_source_voice < len(paths):
            all_source_voices_index = list(range(len(paths)))
            selected_source_voices_index = np.random.choice(all_source_voices_index, args.num_source_voice, replace=False)
            paths = [paths[i] for i in selected_source_voices_index]
            print(len(paths))
        w_start = min(max(args.w_start, 0), len(paths))
        w_end =  len(paths) if args.w_end == -1 else args.w_end
        w_end = min(max(w_end, 0), len(paths))
        print(flag_idx+args.start, w_start, w_end)

        # for a_root, a_dir, a_file in os.walk(os.path.join(pre_dir, flag, infer_dir)):
        #     for f in a_file:
        #         if os.path.join(pre_dir, flag, infer_dir, f) not in ([x[1] for x in paths[args.w_start:args.w_end]] + [x[2] for x in paths[args.w_start:args.w_end]] + [x[3] for x in paths[args.w_start:args.w_end]]):
        #             os.remove(os.path.join(a_root, f))
        #             print(os.path.join(a_root, f))
        
        # for wav_path, des_file, des_file_pitch, ppg_path in paths:
        # for wav_path, des_file, des_file_pitch, ppg_path in paths[args.w_start:args.w_end]:
        for wav_idx, (wav_path, des_file, des_file_pitch, ppg_path) in enumerate(paths[w_start:w_end]):
            condition_gen = (os.path.exists(des_file) and os.path.exists(des_file_pitch)) if not args.force_gen else False
            if condition_gen:
                print(wav_idx+w_start, des_file, 'exists')
                continue
            if not os.path.exists(ppg_path):
                command = 'python svc_inference_ppg.py -w "{}" -p "{}"'.format(wav_path, ppg_path)
                os.system(command)
            else:
                print(ppg_path, 'ppg exists')
            command = 'python svc_inference.py --config config/maxgan.yaml --model {}/maxgan_g.pth --spk "{}/{}/lora_speaker.npy" \
                --wave "{}" --des "{}" --des_pitch "{}" --ppg "{}"'.format(
                    pre_dir, data_root, flag, wav_path, des_file, des_file_pitch, ppg_path,
                )
            if not args.infer_no_pit:
                if not args.crepe:
                    command += ' --statics "{}/{}/lora_pitch_statics.npy"'.format(data_root, flag)
                else:
                    command += ' --statics "{}/{}/lora_pitch_statics_crepe.npy"'.format(data_root, flag)
            if args.singer_pitch_ave_factor != 0.:
                command += ' --singer_pitch_ave_factor {}'.format(abs(args.singer_pitch_ave_factor) if 'M' in flag else -abs(args.singer_pitch_ave_factor))
            print(command)
            os.system(command)