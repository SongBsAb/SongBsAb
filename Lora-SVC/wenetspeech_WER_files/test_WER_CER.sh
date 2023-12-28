
. ./path.sh || exit 1;

# flag=None
# in_out=in
# my_flag='OpenSinger'

# flag=ppg_attack_tar-psy_backtrack-lr=0_0002
# in_out=in
# my_flag=OpenSinger-ppg_attack_tar-psy_backtrack-lr=0_0002

# flag=ppg_attack_tar-psy_backtrack-lr=0_0002-adver-full-mean-two-loss-psy_scale_backtrack-no-eps
# in_out=out
# my_flag=inference-ppg_attack_tar-psy_backtrack-lr=0_0002-adver-full-mean-two-loss-psy_scale_backtrack-no-eps

# flag=all_singers-10_voices
# in_out=out
# my_flag=inference-all_singers-10_voices

# flag=adver-full-mean-two-loss-psy_scale_backtrack-no-eps
# in_out=out
# my_flag=inference-adver-full-mean-two-loss-psy_scale_backtrack-no-eps

# flag=ppg_attack_tar-psy_backtrack-lr=0_0002-all_singers-10_voices
# in_out=out
# my_flag=inference-ppg_attack_tar-psy_backtrack-lr=0_0002-all_singers-10_voices

# flag=ppg_attack_tar-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps
# in_out=out
# my_flag=inference-ppg_attack_tar-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps

# flag=ppg_attack_tar-hubert-psy_backtrack-lr=0_0002-all_singers-10_voices
# in_out=out
# my_flag=inference-ppg_attack_tar-hubert-psy_backtrack-lr=0_0002-all_singers-10_voices

# flag=ppg_attack_tar-whisper-large-psy_backtrack-lr=0_0002-all_singers-10_voices
# in_out=out
# my_flag=inference-ppg_attack_tar-whisper-large-psy_backtrack-lr=0_0002-all_singers-10_voices

# flag=ppg_attack_tar-hubert-psy_backtrack-lr=0_0002-IRA-all_singers-10_voices
# in_out=out
# my_flag=inference-ppg_attack_tar-hubert-psy_backtrack-lr=0_0002-IRA-all_singers-10_voices

flag=ppg_attack_tar-hubert-psy_backtrack-lr=0_0002-IRA
in_out=in
my_flag=OpenSinger-ppg_attack_tar-hubert-psy_backtrack-lr=0_0002-IRA

#########################################################################################
# flag=ppg_attack_tar-hubert+whisper-large-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps
# in_out=out
# my_flag=inference-ppg_attack_tar-hubert+whisper-large-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps

# flag=all_singers-10_voices
# in_out=out
# my_flag=inference-all_singers-10_voices

# flag=ppg_attack_tar-hubert+whisper-large-psy_backtrack-lr=0_0002
# in_out=in
# my_flag=OpenSinger-ppg_attack_tar-hubert+whisper-large-psy_backtrack-lr=0_0002


python test_WER_CER.py -flag $flag -in_out $in_out

data_file=data/${my_flag}.list
result_dir=results
mkdir -p $result_dir
result_file=$result_dir/text-${my_flag}

mode=attention_rescoring
dir=../../../20220506_u2pp_conformer_exp
decode_checkpoint=$dir/final.pt
dict=$dir/units.txt
ctc_weight=0.5
# reverse_weight=0.0
reverse_weight=0.3
python wenet/bin/recognize.py --gpu 0 \
    --mode $mode \
    --config $dir/train.yaml \
    --data_type "raw" \
    --test_data $data_file \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 1 \
    --penalty 0.0 \
    --dict $dict \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_file $result_file \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

if [ "$my_flag" != "OpenSinger" ] 
then
    python tools/compute-wer.py --char=1 --v=1 \
        $result_dir/text-OpenSinger $result_dir/text-${my_flag} > $result_dir/wer-${my_flag}
fi