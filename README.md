
# Source Code for SongBsAb
## Instructions
1.  Download OpenSinger and NUS-CMS-48 datasets and place them at ```./OpenSinger``` and ```./nus-smc-corpus_48```

2. Change the working directory. 
- ```cd Lora-SVC```

3. Build the environment by following the instructions at ```setup.sh```

4. Process the dataset. 
- ```python preprocess_dataset_opensinger.py``` or ```python preprocess_dataset_nus_cms_48.py```

5. Obtain the undefended output singing voices.  
- ```python few_shot_svc.py --type all_singers-10_voices --source_type '' -limit_target_spk -limit_source_voice -num_source_voice 100 -dataset opensinger```

6. Compute the target speaker similarity of undefended output singing voices. 
- ```python cal_speaker_sim_undefended.py --start 0 --end 76 --attack_flag all_singers-10_voices --attack_flag_2 all_singers-10_voices resnet18_veri```

7. Select input source and target singing voices. 
- ```python select_target_source.py```

8. Generate adversarial examples for input target singing voices.  
- ```python gen_adv_psy_scale_Adam_backtrack_non_joint.py --start 0 --end 10 --epsilon 0.1 --max_iter 1000 lora_LSTM```

9. Generate adversarial examples for input source singing voices.  
- ```python source_ppg_attack_tar_psy_backtrack.py --start 0 --end 1000 "whisper"```

10. Obtain the defended output singing voices (dual prevention).  
- ```python few_shot_svc.py -start 0 -end 12 -type adver-full-two-loss-psy_scale_backtrack-no-eps -source_type ppg_attack_tar-whisper-psy_backtrack-lr=0_0002 -dataset opensinger -w_start 0 -w_end 10000```

11. Compute the target speaker similarity of defended output singing voices. 
- ```python cal_speaker_sim.py --start 0 --end 76 --attack_flag adver-full-two-loss-psy_scale_backtrack-no-eps --attack_flag_2 ppg_attack_tar-whisper-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps resnet18_veri```

12. Compute the lyric word error rate of defended output singing voices. 
- ```cd wenet/examples/wenetspeech/s0```
- ```sh test_WER_CER_args.sh ppg_attack_tar-whisper-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps out inference-ppg_attack_tar-whisper-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps```

13. Compute the lyric word error rate of undefended output singing voices. 
- ```sh test_WER_CER_args.sh all_singers-10_voices out inference-all_singers-10_voices```