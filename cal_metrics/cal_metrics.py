import os
import argparse
import librosa
import numpy as np
from compute_metrics import compute_metrics
from rich.progress import track

def get_dataset_filelist(h):
    with open(h.input_test_file, 'r', encoding='utf-8') as fi:
        indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    return indexes

def main(h):
    indexes = get_dataset_filelist(h)
    num = len(indexes)
    print(num)
    metrics_total = np.zeros(6)
    for index in track(indexes):
        clean_wav = os.path.join(h.clean_wav_dir, index + '.wav')
        noisy_wav = os.path.join(h.noisy_wav_dir, index + '.wav')
        clean, sr = librosa.load(clean_wav, h.sampling_rate)
        noisy, sr = librosa.load(noisy_wav, h.sampling_rate)

        metrics = compute_metrics(clean, noisy, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2], 
          'covl: ', metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_rate', default=16000)
    parser.add_argument('--input_test_file', default='dataset_se/test.txt')
    parser.add_argument('--clean_wav_dir', default='dataset_se/testset_clean')
    parser.add_argument('--noisy_wav_dir', default='generated_files/MP-SENet')

    h = parser.parse_args()

    main(h)