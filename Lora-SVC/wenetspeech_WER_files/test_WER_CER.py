
import os

import argparse
paser = argparse.ArgumentParser()
paser.add_argument('-flag', '--flag', default='None')
paser.add_argument('-in_out', '--in_out', default='in')

args = paser.parse_args()

print(args.flag, args.in_out)

if args.in_out == 'in':
    if args.flag is None or args.flag == 'None':
        args.flag = ''
        print(args.flag)
    else:
        args.flag = f'-{args.flag}'
    root = f'../../../../storage/OpenSinger{args.flag}'
    my_flag = f'OpenSinger{args.flag}'
elif args.in_out == 'out':
    if args.flag is None or args.flag == 'None':
        args.flag = 'all_singers-10_voices'
    my_flag = f'inference-{args.flag}'
    root = '../../../../storage/model_pretrain'

import yaml
des_file = '../../../../select-target_speakers-source_speeches-des.yaml'
print(os.path.abspath(des_file))
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

utt2spks = {}
for utt in utts:
    utt2spks[utt] = []
    for spk in spks_keys:
        if utt in spk_2_utt[spk]:
            utt2spks[utt] += [spk]

if args.in_out == 'in' and (args.flag == '' or args.flag is None):
    os.makedirs('results', exist_ok=True)
    gt_text_file = 'results/text-OpenSinger'
    w_m = open(gt_text_file, 'w')

os.makedirs('data', exist_ok=True)
data_file = f'data/{my_flag}.list'
w1 = open(data_file, 'w')
import json
ori_root = '../../../../storage/OpenSinger'
cnt = 0
for g in ['WomanRaw', 'ManRaw']:
    g_dir = os.path.join(ori_root, g)
    for x in sorted(os.listdir(g_dir)):
        if len(x.split('_')) != 2:
            continue
        flag_2 = '{}_{}'.format('M' if g == 'ManRaw' else 'F', x.split('_')[0])
        gender_flag = 'M' if g == 'ManRaw' else 'F'
        for name in sorted(os.listdir(os.path.join(g_dir, x))):
            utt_id = gender_flag + '_' + name
            if 'wav' not in name:
                continue
            if 'backtrack' in name:
                continue
            if utt_id not in utts:
                continue

            ori_path = os.path.join(g_dir, x, name)
            if not os.path.exists(ori_path):
                continue
            
            if args.in_out == 'in' and (args.flag == '' or args.flag is None):
                text_file = os.path.join(ori_root, g, x, name.replace('.wav', '.txt'))
                with open(text_file, 'r') as r_2:
                    gt_text = r_2.readlines()[0]
                    
            if args.in_out == 'in':
                adver_root = root
                adver_path = os.path.join(adver_root, g, x, name)
                print(adver_path)
                assert os.path.exists(adver_path) 
                if os.path.exists(adver_path):
                    for target_spk in utt2spks[utt_id]:
                        my_dict = {"key": str(cnt), "wav": adver_path, "txt": "我不知道"}
                        line = json.dumps(my_dict, ensure_ascii=False) + '\n'
                        w1.write(line)
                        print(line)
                        cnt += 1

                        if args.in_out == 'in' and (args.flag == '' or args.flag is None):
                            line = f'{cnt-1} {gt_text}\n'
                            w_m.write(line)
        
            else:
                for target_spk in utt2spks[utt_id]:
                    out_root = root
                    adver_out_path = os.path.join(out_root, target_spk, my_flag, utt_id)
                    
                    if not os.path.exists(adver_out_path):
                        print('not exists', adver_out_path)
                    assert os.path.exists(adver_out_path) 
                    if os.path.exists(adver_out_path):
                        my_dict = {"key": str(cnt), "wav": adver_out_path, "txt": "我不知道"}
                        line = json.dumps(my_dict, ensure_ascii=False) + '\n'
                        w1.write(line)
                        print(line)
                        cnt += 1
w1.close()
if args.in_out == 'in' and (args.flag == '' or args.flag is None):
    w_m.close()