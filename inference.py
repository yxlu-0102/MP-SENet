from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
from re import S
import torch
import librosa
import numpy as np
from env import AttrDict
from datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import MPNet
import soundfile as sf
from tqdm import tqdm

h = None
device = None

def wsola_chunked_processing(audio, sr, chunk_size, hop_size, mod_func):
    # Check if chunk_size is larger than the audio length
    if chunk_size >= len(audio):
        # Process the entire audio in one go
        output = mod_func(audio).squeeze()
    else:
        # Initialize the output array
        output = np.array([], dtype=audio.dtype)

        # Initialize the start point of the first chunk
        start = 0

        # Calculate total number of chunks
        total_chunks = (len(audio) - hop_size) // (chunk_size - hop_size) + 1

        # WSOLA chunked processing loop with progress bar
        with tqdm(total=total_chunks, desc="Processing audio chunks") as pbar:
            while start < len(audio)-hop_size:
                # Calculate the end point of the current chunk
                end = min(start + chunk_size, len(audio))

                # Get the current chunk and apply the modifying function
                chunk = audio[start:end]
                modified_chunk = mod_func(chunk).squeeze()

                if start == 0:
                    # For the first chunk, append the entire modified chunk
                    output = np.append(output, modified_chunk)
                else:
                    # Find the best overlapping point using cross-correlation
                    overlap = output[-hop_size:]
                    correlation = np.correlate(modified_chunk[:hop_size*2], overlap, mode='valid')
                    best_offset = np.argmax(correlation)

                    # Overlap and add using the best offset
                    crossfade = np.linspace(0, 1, hop_size)
                    output[-hop_size:] = output[-hop_size:] * (1 - crossfade) + modified_chunk[best_offset:best_offset+hop_size] * crossfade

                    # Append the non-overlapping part of the modified chunk to the output
                    output = np.append(output, modified_chunk[best_offset+hop_size:])

                # Move to the next chunk
                start = end - hop_size

                # Update progress bar
                pbar.update(1)

    # Normalize the output
    output /= np.max(np.abs(output))

    return output

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

    model.eval()

    with torch.no_grad():
        noisy_wav, _ = librosa.load(a.input_noisy_wav, sr=h.sampling_rate)

        def denoise(noisy_wav):
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            audio_g = audio_g / norm_factor
            return audio_g.cpu().numpy()

        chunk_size_samples = int(a.chunk_size * h.sampling_rate)
        hop_size_samples = int(a.hop_size * h.sampling_rate)
        
        audio_g = wsola_chunked_processing(
            noisy_wav, sr=h.sampling_rate, chunk_size=chunk_size_samples, hop_size=hop_size_samples, mod_func=denoise
        )

        sf.write(a.output_file, audio_g.squeeze(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wav', required=True, help='Path to the input noisy wav file')
    parser.add_argument('--output_file', required=True, help='Path to the output denoised wav file')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--chunk_size', type=float, default=1.0, help='Chunk size for WSOLA processing in seconds')
    parser.add_argument('--hop_size', type=float, default=0.05, help='Hop size for WSOLA processing in seconds')
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

