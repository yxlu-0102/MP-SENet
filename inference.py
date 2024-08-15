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

h = None
device = None

def wsola_chunked_processing(audio, sr, chunk_size, hop_size, mod_func):
    # Calculate the number of chunks needed for the input audio
    num_chunks = int(np.ceil(len(audio) / hop_size))

    # Initialize the output array
    output = np.array([], dtype=audio.dtype)

    # WSOLA chunked processing loop
    for i in range(num_chunks):
        # Calculate the start and end points of the current chunk
        start = i * hop_size
        end = min(start + chunk_size, len(audio))

        # Get the current chunk and apply the modifying function
        chunk = audio[start:end]
        modified_chunk = mod_func(chunk)

        if i == 0:
            # For the first chunk, append the entire modified chunk
            output = np.append(output, modified_chunk)
        else:
            # Find the most similar chunk in the input audio for the overlapping region
            overlap_start = start - hop_size
            overlap_end = start

            best_match = None
            best_distance = float('inf')
            for j in range(max(0, i-5), i):  # Look at the 5 previous chunks
                # Calculate the start and end points of the comparison chunk
                start_j = j * hop_size
                end_j = min(start_j + chunk_size, len(audio))

                # Get the overlapping region of the comparison chunk
                overlap_chunk_j = audio[max(start_j, overlap_start):min(end_j, overlap_end)]

                # Compute the distance between the overlapping regions
                distance = np.sum((output[-hop_size:] - overlap_chunk_j) ** 2)

                # Update the best match if necessary
                if distance < best_distance:
                    best_match = overlap_chunk_j
                    best_distance = distance

            # Overlap and add the best matching chunk to the output
            crossfade = np.linspace(0, 1, hop_size)
            output[-hop_size:] = output[-hop_size:] * (1 - crossfade) + best_match * crossfade

            # Append the non-overlapping part of the modified chunk to the output
            output = np.append(output, modified_chunk[hop_size:])

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

        audio_g = wsola_chunked_processing(
            noisy_wav, sr=h.sampling_rate, chunk_size=h.sampling_rate, hop_size=1024, mod_func=denoise
        )

        sf.write(a.output_file, audio_g.squeeze(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wav', required=True, help='Path to the input noisy wav file')
    parser.add_argument('--output_file', required=True, help='Path to the output denoised wav file')
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

