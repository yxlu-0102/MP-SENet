from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append("..")
import glob
import os
import argparse
import json
from re import S
import numpy as np
import torch
import librosa
from env import AttrDict
from dataset import mag_pha_stft, mag_pha_istft
from models.model import MPNet
import soundfile as sf
from rich.progress import track

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(a):
    model = MPNet(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    test_indexes = os.listdir(a.input_noisy_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    max_chunk_duration = 10  # 10 seconds cutoff
    chunk_size = int(max_chunk_duration * h.sampling_rate)
    with torch.no_grad():
        for index in track(test_indexes):
            noisy_wav, _ = librosa.load(os.path.join(a.input_noisy_wavs_dir, index), sr=h.sampling_rate)
            
            # Always process in chunks
            num_chunks = int(np.ceil(len(noisy_wav) / chunk_size))
            audio_chunks = []
            
            for i in range(num_chunks):
                start = int(i * chunk_size)
                end = int(min((i + 1) * chunk_size, len(noisy_wav)))
                chunk = noisy_wav[start:end]
                
                chunk = torch.FloatTensor(chunk).to(device)
                norm_factor = torch.sqrt(len(chunk) / torch.sum(chunk ** 2.0)).to(device)
                chunk = (chunk * norm_factor).unsqueeze(0)

                noisy_amp, noisy_pha, _ = mag_pha_stft(chunk, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                amp_g, pha_g, _ = model(noisy_amp, noisy_pha)
                chunk_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

                chunk_g = chunk_g / norm_factor
                audio_chunks.append(chunk_g.squeeze().cpu().numpy())
            # Concatenate all chunks (will be one chunk if audio was < 10s)
            audio_g = np.concatenate(audio_chunks)
            
            output_file = os.path.join(a.output_dir, index)
            sf.write(output_file, audio_g, h.sampling_rate, 'PCM_16')

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default='VoiceBank+DEMAND/testset_noisy')
    parser.add_argument('--output_dir', default='../generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
