import librosa
import soundfile as sf
import os
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
import torch
import torchaudio.functional as aF

def flac2wav(wav):
    y, sr = librosa.load(wav, sr=48000, mono=True)
    file_id = os.path.split(wav)[-1].split('_mic')[0]
    if file_id in timestamps:
        start, end = timestamps[file_id]
        start = start - min(start, int(0.1 * sr))
        end = end + min(len(y) - end, int(0.1 * sr))
        y = y[start: end]

    # y = torch.FloatTensor(y).unsqueeze(0)
    # y = aF.resample(y, orig_freq=sr, new_freq=22050).squeeze().numpy()

    os.makedirs(os.path.join('wav48_silence_trimmed', wav.split(os.sep)[-2]), exist_ok=True)

    wav_path = os.path.join('wav48_silence_trimmed', wav.split(os.sep)[-2], file_id +'.wav')

    sf.write(wav_path, y, 48000, 'PCM_16')
    del y
    return


if __name__=='__main__':

    base_dir = 'wav48_origin'

    wavs = glob(os.path.join(base_dir, '*/*mic1.flac'))
    sampling_rate = 48000

    timestamps = {}
    path_timestamps = 'vctk-silence-labels/vctk-silences.0.92.txt'
    with open(path_timestamps, 'r') as f:
        timestamps_list = f.readlines()
    for line in timestamps_list:
        timestamp_data = line.strip().split(' ')
        if len(timestamp_data) == 3:
            file_id, t_start, t_end = timestamp_data
            t_start = int(float(t_start) * sampling_rate)
            t_end = int(float(t_end) * sampling_rate)
            timestamps[file_id] = (t_start, t_end)

    pool = mp.Pool(processes = 8)
    with tqdm(total = len(wavs)) as pbar:
        for _ in tqdm(pool.imap_unordered(flac2wav, wavs)):
            pbar.update()
