
import os
import sys
python_path = sys.executable
python_prefix = '/'.join(python_path.split('/')[:-2])
wav2vec2_file = f'{python_prefix}/lib/python3.8/site-packages/s3prl/upstream/wav2vec2/wav2vec2_model.py'
wav2vec2_file_back = f'{python_prefix}/lib/python3.8/site-packages/s3prl/upstream/wav2vec2/wav2vec2_model_backup.py'
os.system(f'cp {wav2vec2_file} {wav2vec2_file_back}')
os.system(f'cat /dev/null > {wav2vec2_file}')
with open(wav2vec2_file_back, 'r') as r, open(wav2vec2_file, 'w') as w:
    for line in r.readlines():
        if 'with torch.no_grad():' in line:
            line = line.replace('with torch.no_grad():', 'if True:')
        w.write(line)