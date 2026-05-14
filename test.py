import numpy as np
import scipy as spy
import torch
import sklearn as sk
import pandas as pd
import librosa
import pathlib
SAMPLE_RATE    = 48_000
SEGMENT_SEC    = 0.25       # window length per impact
MIN_GAP_SEC    = 0.10       # minimum spacing between detected impacts
BANDPASS_LOW   = 3_000.0    # Hz — lower edge of region of interest
BANDPASS_HIGH  = 5_000.0    # Hz — upper edge of region of interest

# FFT
N_FFT      = 8192   # 5.9 Hz/bin — critical for resolving resonant peaks
HOP_LENGTH = 512    # ~10 ms
N_MELS     = 128

# Resonance analysis
N_PEAKS          = 7     # top resonant modes to track
PEAK_PROMINENCE  = 0.05  # min peak prominence (fraction of max)

# Training
N_FOLDS              = 3
BATCH_SIZE           = 32
MAX_EPOCHS           = 150
LEARNING_RATE        = 1e-3
WEIGHT_DECAY         = 5e-5
DROPOUT              = 0.2
EARLY_STOP_PATIENCE  = 20
LABEL_SMOOTHING      = 0.1

print('Hyperparameters configured.')
print(f'FFT resolution : {SAMPLE_RATE / N_FFT:.1f} Hz/bin')
print(f'Bins in 3-5 kHz: {int((BANDPASS_HIGH - BANDPASS_LOW) / (SAMPLE_RATE / N_FFT))}')

for i in pathlib.Path('train').iterdir():
    si = int(str(i)[6:7])
    
