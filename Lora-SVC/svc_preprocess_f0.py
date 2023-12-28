import os
import numpy as np
import librosa
import pyworld
import torch
import torchcrepe

def compute_f0(path):
    x, sr = librosa.load(path, sr=16000)
    assert sr == 16000
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=900,
        frame_period=1000 * 160 / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs=16000)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return f0


def compute_f0_nn_no_post(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = torch.tensor(np.copy(audio))[None]
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 1000
    # Select a model capacity--one of "tiny" or "full"
    model = "full"
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 512
    # Compute pitch using first gpu
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    periodicity = np.repeat(periodicity, 2, -1)  # 320 -> 160 * 2
    # CREPE was not trained on silent audio. some error on silent need filter.
    # periodicity = torchcrepe.filter.median(periodicity, 9)
    # pitch = torchcrepe.filter.mean(pitch, 9)
    # pitch[periodicity < 0.1] = 0
    pitch = pitch.squeeze(0)
    return pitch.detach().cpu().numpy()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data_svc", help="Directory path containing input audio files.")
    parser.add_argument("--crepe", action='store_true', default=False)
    args = parser.parse_args()

    filelist_dir = os.path.join(args.root, "filelists")
    os.makedirs(filelist_dir, exist_ok=True)
    files = open("{}/train.txt".format(filelist_dir), "w", encoding="utf-8")
    # files_eval = open("{}/eval.txt".format(filelist_dir), "w", encoding="utf-8")

    rootPath = "{}/waves/".format(args.root)
    outPath = "{}/pitch/".format(args.root) if not args.crepe else "{}/pitch_crepe/".format(args.root)
    os.makedirs(outPath, exist_ok=True)

    infos = []
    for file in os.listdir(f"./{rootPath}"):
        if file.endswith(".wav"):
            file = file[:-4]
            wav_path = f"./{rootPath}//{file}.wav"
            featur_pit = compute_f0(wav_path) if not args.crepe else compute_f0_nn_no_post(wav_path, "cuda:0")

            np.save(
                f"{outPath}//{file}.nsf",
                featur_pit,
                allow_pickle=False,
            )

            path_spk = "{}/lora_speaker.npy".format(args.root)
            path_wave = f"{args.root}/waves/{file}.wav"
            path_pitch = f"{args.root}/pitch/{file}.nsf.npy"
            path_whisper = f"{args.root}/whisper/{file}.ppg.npy"
            # print(
            #     f"{path_wave}|{path_pitch}|{path_whisper}|{path_spk}",
            #     file=files,
            # )
            infos.append(f"{path_wave}|{path_pitch}|{path_whisper}|{path_spk}")

    # assert len(infos) >= 15, "Less than 15 files"
    # for x in infos[:-5]:
    for x in infos:
        print(x, file=files,)
    # for x in infos[-5:]:
    #     print(x, file=files_eval,)
    files.close()
    # files_eval.close()