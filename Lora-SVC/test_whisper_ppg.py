import os
import numpy as np
import argparse
import torch
import torchaudio

from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


def load_model(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def pred_ppg(whisper: Whisper, wavPath, ppgPath):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = audln // 320
    # audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).to(whisper.device)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppgln,] # [length, dim=1024]
        print(ppg.shape)
        np.save(ppgPath, ppg, allow_pickle=False)

def pred_ppg_t(whisper: Whisper, audio):
    # audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = audln // 320
    # audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).to(whisper.device)
    ppg = whisper.encoder(mel.unsqueeze(0)).squeeze()
    ppg = ppg[:ppgln,] # [length, dim=1024]
    # print(ppg.shape)
    return ppg


def pred_ppg_infer_t(whisper: Whisper, audio):
    audln = audio.shape[0]
    ppg_a = []
    idx_s = 0
    while (idx_s + 25 * 16000 < audln):
        short = audio[idx_s:idx_s + 25 * 16000]
        idx_s = idx_s + 25 * 16000
        ppgln = 25 * 16000 // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(whisper.device)
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze()
        ppg = ppg[:ppgln,]  # [length, dim=1024]
        # print(ppg.shape)
        ppg_a.extend(ppg)
    if (idx_s < audln):
        short = audio[idx_s:audln]
        ppgln = (audln - idx_s) // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(whisper.device)
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze()
        ppg = ppg[:ppgln,]  # [length, dim=1024]
        # print(ppg.shape)
        ppg_a.extend(ppg)
    # print(len(ppg_a), torch.stack(ppg_a).shape)
    return torch.stack(ppg_a)

def pred_ppg_infer_t_batch(whisper: Whisper, audio):
    audln = audio.shape[1]
    ppg_a = []
    idx_s = 0
    while (idx_s + 25 * 16000 < audln):
        short = audio[:, idx_s:idx_s + 25 * 16000]
        idx_s = idx_s + 25 * 16000
        ppgln = 25 * 16000 // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(whisper.device)
        ppg = whisper.encoder(mel)
        ppg = ppg[:, :ppgln, :]  # [length, dim=1024]
        # print(ppg.shape)
        ppg_a.append(ppg)
    if (idx_s < audln):
        short = audio[:, idx_s:audln]
        ppgln = (audln - idx_s) // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(whisper.device)
        ppg = whisper.encoder(mel)
        ppg = ppg[:, :ppgln, :]  # [length, dim=1024]
        # print(ppg.shape)
        ppg_a.append(ppg)
    # print(len(ppg_a), torch.stack(ppg_a).shape)
    return torch.cat(ppg_a)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    args = parser.parse_args()
    print(args.wav)
    wavPath = args.wav

    whisper = load_model(os.path.join("whisper_pretrain", "medium.pt"))

    # wavPath = args.wav
    # # audio = load_audio(wavPath).cuda()
    # audio = torchaudio.load(wavPath)[0].squeeze(0)
    # audio.requires_grad = True
    # ppg = pred_ppg_t(whisper, audio)
    # ppg.backward(torch.ones_like(ppg))
    # print(audio.grad, audio.grad.shape)

    #############################################################
    # import whisper_openai

    # model = whisper_openai.load_model("medium")

    # # load audio and pad/trim it to fit 30 seconds
    # audio = whisper_openai.load_audio(wavPath)
    # print(audio.max(), audio.min(), audio.shape)
    # audio = whisper_openai.pad_or_trim(audio)
    # print(audio.shape)

    # # make log-Mel spectrogram and move to the same device as the model
    # mel = whisper_openai.log_mel_spectrogram(audio).to(model.device)

    # # detect the spoken language
    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # # decode the audio
    # options = whisper_openai.DecodingOptions()
    # result = whisper_openai.decode(model, mel, options)

    # print(result)

    # # print the recognized text
    # print(result.text)


    ##################################
    # ### local mel
    # audio_ori = torchaudio.load(wavPath)[0].cuda()
    # audio_ori.requires_grad = True
    # audio = audio_ori.squeeze(0)
    # audln = audio.shape[0]
    # ppgln = audln // 320
    # audio = pad_or_trim(audio)
    # mel = log_mel_spectrogram(audio)
    # # mel = mel.detach().cpu().numpy()
    # print(mel, mel.shape)

    # ### hugging face mel
    # import librosa
    # from transformers import WhisperFeatureExtractor
    # model_type = 'medium'
    # feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-{}".format(model_type))

    # audio, sr_original = torchaudio.load(wavPath)
    # print('fs:', sr_original)
    # audio = audio.squeeze(0).numpy()
    # sr_target = 16000
    # # Resample to target sample rate
    # # audio = librosa.resample(audio, orig_sr=sr_original, target_sr=sr_target)
    # mel2 = feature_extractor(audio, sampling_rate=sr_target).input_features[0]

    # print(mel, mel.shape)
    # # mel2 = mel2[..., :mel.shape[-1]]
    # print(mel2, mel2.shape)
    # # print(mel2[..., :mel.shape[-1]], mel2.shape)
    # print((mel == mel2).all())

    # for x, y in zip(mel, mel2):
    #     print(abs(x-y).sum())

    text = '情深深雨濛濛'
    from transformers import WhisperTokenizer
    model_type = 'medium'
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-{}".format(model_type), language="zh", task="transcribe")

    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()

    do_normalize_text = True
    input_str = normalizer(text).strip() if do_normalize_text else text
    labels = tokenizer(input_str).input_ids
    print(labels)

    from transformers import WhisperForConditionalGeneration
    model_type = 'medium'
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-{}".format(model_type))
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # model.config.dropout = 0.1
    model.config.use_cache = False

    from transformers import Seq2SeqTrainingArguments
    output_dir = './test'
    per_device_train_batch_size = 1
    max_steps = 1
    per_device_eval_batch_size = 1
    training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    # max_steps=math.ceil(tol_train * num_epoch / per_device_train_batch_size / num_gpu),
    max_steps=max_steps,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=per_device_eval_batch_size,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="es-test_wer",
    greater_is_better=False,
    push_to_hub=False,
    seed=42,
    data_seed=42,
)
    print(training_args.label_smoothing_factor)
    from transformers import Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    )

    trainer.model.eval()
    inputs = {'input_features': mel.unsqueeze(0), 'labels': torch.tensor(labels).unsqueeze(0).cuda()}
    out = trainer.compute_loss(model, inputs, return_outputs=False)
    print(out)
    out.backward()
    print(audio_ori.grad, audio_ori.grad.shape)

    from transformers import WhisperProcessor
    model_type = 'medium'
    processor = WhisperProcessor.from_pretrained("openai/whisper-{}".format(model_type), language="zh", task="transcribe")
    generated_ids = model.generate(inputs=mel.unsqueeze(0), task='transcribe', language='zh', is_multilingual=True)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(transcription)

    # audio = torchaudio.load(wavPath)[0].squeeze(0).cuda()
    # audio.requires_grad = True
    # ppg = pred_ppg_infer_t(whisper, audio)
    # print(ppg.shape)
    # ppg.backward(torch.ones_like(ppg))
    # print(audio.grad, audio.grad.shape)