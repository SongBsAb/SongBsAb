
. ./path.sh || exit 1;

# flag=None
# in_out=in
# my_flag='NUS-CMS-48-2'

# flag=None
# in_out=out
# my_flag=inference

# flag=ppg_attack_tar-psy_backtrack-lr=0_0002
# in_out=in
# my_flag=NUS-CMS-48-2-ppg_attack_tar-psy_backtrack-lr=0_0002

# flag=ppg_attack_tar-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps
# in_out=out
# my_flag=inference-ppg_attack_tar-psy_backtrack-lr=0_0002-adver-full-two-loss-psy_scale_backtrack-no-eps

# flag=ppg_attack_tar-psy_backtrack-lr=0_0002
# in_out=out
# my_flag=inference-ppg_attack_tar-psy_backtrack-lr=0_0002

flag=adver-full-two-loss-psy_scale_backtrack-no-eps
in_out=out
my_flag=inference-adver-full-two-loss-psy_scale_backtrack-no-eps


python test_WER_CER.py -flag $flag -in_out $in_out

data_file=data/${my_flag}.list
result_dir=results
mkdir -p $result_dir
result_file=$result_dir/text-${my_flag}


mode=attention_rescoring
dir=../../../20210728_u2pp_conformer_exp
bpemodel=$dir/train_xl_unigram5000
decode_checkpoint=$dir/final.pt
dict=$dir/units.txt
ctc_weight=0.5
gpu_id=0
python wenet/bin/recognize.py --gpu $gpu_id \
        --mode $mode \
        --config $dir/train.yaml \
        --data_type "raw" \
        --bpe_model $bpemodel.model \
        --test_data $data_file \
        --checkpoint $decode_checkpoint \
        --beam_size 20 \
        --batch_size 1 \
        --penalty 0.0 \
        --dict $dict \
        --result_file $result_file \
        --ctc_weight $ctc_weight \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

if [ "$my_flag" != "NUS-CMS-48-2" ] 
then
    python tools/compute-wer.py --char=1 --v=1 \
        $result_dir/text-NUS-CMS-48-2 $result_dir/text-${my_flag} > $result_dir/wer-${my_flag}
fi

tail -n 8 $result_dir/wer-${my_flag}