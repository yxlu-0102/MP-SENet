# MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra
### Ye-Xin Lu, Yang Ai, Zhen-Hua Ling
In our paper, we proposed MP-SENet: a TF-domain monaural SE model with parallel magnitude and phase spectra denoising.
We provide our implementation and pretrained models as open source in this repository.

**Abstract:** 
This paper proposes MP-SENet, a novel Speech Enhancement Network which directly denoises Magnitude and Phase spectra in parallel. The proposed MP-SENet adopts a codec architecture in which the encoder and decoder are bridged by convolution-augmented transformers. The encoder aims to encode time-frequency representations from the input noisy magnitude and phase spectra. The decoder is composed of parallel magnitude mask decoder and phase decoder, directly recovering clean magnitude spectra and clean-wrapped phase spectra by incorporating learnable sigmoid activation and parallel phase estimation architecture, respectively. Multi-level losses defined on magnitude spectra, phase spectra, short-time complex spectra, and time-domain waveforms are used to train the MP-SENet model jointly. Experimental results show that our proposed MP-SENet achieves a PESQ of 3.50 on the public VoiceBank+DEMAND dataset and outperforms existing advanced SE methods.

Visit our [demo website](http://yxlu-0102.github.io/mpsenet-demo) for audio samples.
