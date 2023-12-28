
# SVC model
conda create -n lora-svc python=3.8
conda activate lora-svc
pip install -r requirements.txt

# wenet; using for calculating WER
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
pip install -r requirements.txt
pre-commit install  # for clean and tidy code
cd ../
cp -r wenetspeech_WER_files wenet/examples/wenetspeech/
mv wenet/examples/wenetspeech/wenetspeech_WER_files/* wenet/examples/wenetspeech/
cp -r gigaspeech_WER_files wenet/examples/gigaspeech/
mv wenet/examples/gigaspeech/gigaspeech_WER_files/* wenet/examples/gigaspeech/

# s3prl; used for lyric encoder wav2vec2 and decoar2
pip install s3prl
python solve_no_grad.py

pip install pydub