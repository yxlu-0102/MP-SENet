import os
import argparse
import librosa
import pysepm
import numpy as np
from pesq import pesq
from rich.progress import track


def sisdr(x, s, remove_dc=True):
    """
    Compute SI-SDR
    x: extracted signal
    s: reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))


def main(h):
    indexes = os.listdir(h.noisy_wav_dir)
    metrics = {'pesq_wb':[], 'pesq_nb':[], 'stoi':[], 'sisdr': [], 'apd': []}

    for index in track(indexes):
        
        noisy_wav = os.path.join(h.noisy_wav_dir, index)
        clean_wav = os.path.join(h.clean_wav_dir,index)

        clean, _ = librosa.load(clean_wav, sr=h.sampling_rate)
        noisy, _ = librosa.load(noisy_wav, sr=h.sampling_rate)
        length = min(len(clean), len(noisy))
        clean = clean[0: length]
        noisy = noisy[0: length]

        pesq_wb_score = pesq(fs=h.sampling_rate, ref=clean, deg=noisy, mode="wb")
        pesq_nb_score = pesq(fs=h.sampling_rate, ref=clean, deg=noisy, mode="nb")
        stoi_score = pysepm.stoi(clean, noisy, h.sampling_rate)
        sisdr_score = sisdr(noisy, clean)

        metrics['pesq_wb'].append(pesq_wb_score)
        metrics['pesq_nb'].append(pesq_nb_score)
        metrics['stoi'].append(stoi_score)
        metrics['sisdr'].append(sisdr_score)

    pesq_wb_mean = np.mean(metrics['pesq_wb'])
    pesq_nb_mean = np.mean(metrics['pesq_nb'])
    stoi_mean = np.mean(metrics['stoi'])
    sisdr_mean = np.mean(metrics['sisdr'])


    print('PESQ_WB: {:.3f}'.format(pesq_wb_mean))
    print('PESQ_NB: {:.3f}'.format(pesq_nb_mean))
    print('STOI: {:.3f}'.format(stoi_mean*100))
    print('SI-SDR: {:.3f}'.format(sisdr_mean))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_rate', default=16000)
    parser.add_argument('--clean_wav_dir', required=True)
    parser.add_argument('--noisy_wav_dir', required=True)

    h = parser.parse_args()

    main(h)