
# select target singers and source voices based on sim of normal out (without ppg); lora-16k & OpenSinger; 
# resnet18_veri

import numpy as np

normal_out_score_file = 'txt_files/resnet18_veri/normal-out_sim.txt'
source_score_file = 'txt_files/resnet18_veri/source_sim.txt'

scores_dict = dict()

with open(normal_out_score_file, 'r') as r:
    lines = r.readlines()
    for line in lines:
        line = line[:-1]
        if 'pitch.wav' in line:
            continue
        spk, utt, score = line.split(' ')
        score = float(score)
        key = f'{spk}@{utt}'
        if key not in scores_dict.keys():
            scores_dict[key] = []
        scores_dict[key] += [score]
        

with open(source_score_file, 'r') as r:
    lines = r.readlines()
    for line in lines:
        line = line[:-1]
        if 'pitch.wav' in line:
            continue
        spk, utt, score = line.split(' ')
        score = float(score)
        key = f'{spk}@{utt}'
        if key not in scores_dict.keys():
            scores_dict[key] = []
        scores_dict[key] += [score]

key2differ = dict()
spk2scores = dict()
for k, v in scores_dict.items():
    if len(v) != 2:
        print(k, len(v))
    else:
        print(k, v[0], v[1])
    key2differ[k] = (v[0], v[1])

key_differ = sorted(key2differ.items(), key=lambda x: x[1][0], reverse=True)
# print(key_differ)
# print(key_differ[:1000])

cover_spks = set()
spk2cnt = dict()
spk2utts = dict()
for x in key_differ[:1000]:
    key = x[0]
    spk = key.split('@')[0]
    cover_spks.add(spk)
    if spk not in spk2cnt.keys():
        spk2cnt[spk] = 0
        spk2utts[spk] = []
    spk2cnt[spk] += 1
    spk2utts[spk] += [key]
print(cover_spks, len(cover_spks))
print()
print(spk2cnt)
print()
print(spk2utts)

normals = []
sources = []
for x in key_differ[:1000]:
    key = x[0]
    spk = key.split('@')[0]
    normal = x[1][0]
    source = x[1][1]
    normals.append(normal)
    sources.append(source)

from matplotlib import pyplot as plt
plt.hist(normals, bins=100, label='undefended output')
plt.hist(sources, bins=100, label='source')
plt.legend()
plt.show()
print(np.mean(normals), np.mean(sources), np.mean(normals) - np.mean(sources))