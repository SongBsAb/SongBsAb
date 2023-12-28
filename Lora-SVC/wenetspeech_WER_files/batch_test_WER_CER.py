
# #### robustness
# import sys
# sys.path.append('../../../../')
# from defense.defense import parser_defense
# import os

# defenses = []
# defense_params = []
# for q in [0, 4, 9]:
#     defenses.append('MP3_V')
#     defense_params.append(q)

# for q in [1, 3, 5]:
#     defenses.append('AAC_V')
#     defense_params.append(q)

# for q in [10, 20, 30]:
#     defenses.append('AT')
#     defense_params.append(q)

# flags = []
# my_flags = []
# for defense, defense_param in zip(defenses, defense_params):
#     defense_functions_flags, defense_name = parser_defense([defense], [str(defense_param)], [0], 'sequential')
#     defense_name = defense_name.replace('&', '#')
#     type = f'adver-full-two-loss-psy_scale_backtrack-no-eps-{defense_name}'
#     source_type = f'ppg_attack_tar-psy_backtrack-lr=0_0002-{defense_name}'

#     attack_flag_2 = f'{source_type}-{type}'

#     flags += [attack_flag_2]
#     my_flags += [f'inference-{attack_flag_2}']

# for defense, defense_param in zip(defenses, defense_params):
#     defense_functions_flags, defense_name = parser_defense([defense], [str(defense_param)], [0], 'sequential')
#     defense_name = defense_name.replace('&', '#')
#     type = f'all_singers-10_voices-{defense_name}'
#     source_type = f'{defense_name}'

#     attack_flag_2 = f'{source_type}-{type}'

#     flags += [attack_flag_2]
#     my_flags += [f'inference-{attack_flag_2}']

# print(flags)
# print(len(my_flags))

# import argparse
# my_parser = argparse.ArgumentParser()
# my_parser.add_argument('start', type=int, default=0)
# my_parser.add_argument('end', type=int, default=len(flags))

# my_args = my_parser.parse_args()

# for flag, my_flag in zip(flags[my_args.start:my_args.end], my_flags[my_args.start:my_args.end]):
#     command = f'sh test_WER_CER_args.sh {flag} out {my_flag}'
#     print(command)
#     os.system(command)


# ## ratio test
# import os

# for i in range(1, 10, 2):

#     flag = f'ppg_attack_tar-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps-pn={i}'
#     my_flag = f'inference-ppg_attack_tar-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps-pn={i}'
#     command = f'sh test_WER_CER_args.sh {flag} out {my_flag}'
#     print(command)
#     os.system(command)

import os
# ## lyric transfer
# system_types = ['whisper-tiny', 'whisper-base', 'whisper-small', 
#                 'wav2vec2', 'decoar2']
# system_types = ['whisper-tiny', 'whisper-base', 'whisper-small', 
#                 'decoar2']
# system_types = ['wav2vec2']
system_types = ['whisper-base']
flags = []
my_flags = []
for system_type in system_types:
    # my_flag = f'inference-ppg_attack_tar-{system_type}-psy_backtrack-lr=0_0002-all_singers-10_voices'
    # my_flags += [my_flag]
    # flag = f'ppg_attack_tar-{system_type}-psy_backtrack-lr=0_0002-all_singers-10_voices'
    # flags += [flag]

    my_flag = f'inference-ppg_attack_tar-{system_type}-psy_backtrack-lr=0_0002-IRA-all_singers-10_voices'
    my_flags += [my_flag]
    flag = f'ppg_attack_tar-{system_type}-psy_backtrack-lr=0_0002-IRA-all_singers-10_voices'
    flags += [flag]

for flag, my_flag in zip(flags, my_flags):
    command = f'sh test_WER_CER_args.sh {flag} out {my_flag}'
    print(command)
    os.system(command)