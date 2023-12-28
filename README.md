
# Source Code for SongBsAb
## Instructions
- Download OpenSinger and NUS-CMS-48 datasets and place them at ```./OpenSinger``` and ```./nus-smc-corpus_48```
- ```cd Lora-SVC```
- Build the environment by following the instructions at ```setup.sh```
- ```python preprocess_dataset_opensinger.py``` or ```python preprocess_dataset_nus_cms_48.py```
- ```python few_shot_svc.py --type all_singers-10_voices --source_type '' -limit_target_spk -limit_source_voice -num_source_voice 100 -dataset opensinger```
- ```python cal_speaker_sim_undefended.py --start 0 --end 76 --attack_flag all_singers-10_voices --attack_flag_2 all_singers-10_voices resnet18_veri```
- ```python select_target_source.py```
- ```python gen_adv_psy_scale_Adam_backtrack_non_joint.py --start 0 --end 10 --epsilon 0.1 --max_iter 1000 lora_LSTM```
- ```python source_ppg_attack_tar_psy_backtrack.py --start 0 --end 1000 "whisper"```
- ```python few_shot_svc.py -start 0 -end 12 -type adver-full-two-loss-psy_scale_backtrack-no-eps -source_type ppg_attack_tar-whisper-psy_backtrack-lr=0_0002 -dataset opensinger -w_start 0 -w_end 10000```
- ```python cal_speaker_sim.py --start 0 --end 76 --attack_flag adver-full-two-loss-psy_scale_backtrack-no-eps --attack_flag_2 ppg_attack_tar-whisper-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps resnet18_veri```