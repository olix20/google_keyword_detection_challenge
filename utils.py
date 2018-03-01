from __future__ import division

from pydub import AudioSegment
import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

import re
from glob import glob
# import matplotlib.pyplot as plt

# import wave

from scipy.io import wavfile
from scipy.signal import stft
import random
import pickle
import numba 

from IPython.lib.display import FileLink

# from librosa import display






def augment_wav(wav,pval=0.5):
    sample_rate = 16000
    L = 1000 #16000  # 1 sec
    
#     adjust speed, with 50% chance
    wav = speed_change(wav,1.+ random.choice([.1,-0.1,0])) #random.uniform(-1, 1)*0.05) if np.random.random() < pval else wav
    
    
    #adjust volume
#     db_adjustment = random.uniform(-1, 1)*10
    wav = wav + random.choice([-10,-5,0,5,10]) #randodb_adjustment if np.random.random() < pval else wav
     
        
    #fill to 1 second
    wav = fill_to_1sec(wav)        
        
    #shift the audio by 10 ms
    shift_length = 100
    if np.random.random() < 0.5: #shift to left
        wav = wav[:L-shift_length]+ AudioSegment.silent(shift_length,frame_rate=sample_rate)
    else: #shift to right
        wav = AudioSegment.silent(shift_length,frame_rate=sample_rate) + wav[shift_length:]
        
        
        
    #blend original file with background noise     
#     if np.random.random() < pval:
    noise = random.choice(silence_files_AS)
    db_delta = (wav.dBFS - noise.dBFS) -10.

    if db_delta< 0: #reduce intensity of loud background; if it's too silent, leave it be
        noise = noise  + db_delta
    wav = wav.overlay(noise)
 
    return wav



def plot_mel(log_S):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=16000, x_axis='time', y_axis='mel')
    plt.title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()


def log_specgram(audio, sample_rate, window_size=20,
				 step_size=10, eps=1e-10):
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	freqs, times, spec = signal.spectrogram(audio,
									fs=sample_rate,
									window='hann',
									nperseg=nperseg,
									noverlap=noverlap,
									detrend=False)
	return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def get_log_mel(S):
	log_S = librosa.power_to_db(S, ref=np.max)
	return log_S


def speed_change(sound, speed=1.0):
	# Manually override the frame_rate. This tells the computer how many
	# samples to play per second
	sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
		"frame_rate": int(sound.frame_rate * speed)
	})


	# slow_sound = speed_change(sound, 0.75)
	# fast_sound = speed_change(sound, 2.0)

	# convert the sound with altered frame rate to a standard frame rate
	# so that regular playback programs will work right. They often only
	# know how to play audio at standard frame rate (like 44.1k)
	return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)


def log_mel(samples,sample_rate=16000,reshape=True,n_mels=128):

		S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=n_mels)

		# Convert to log scale (dB). We'll use the peak power (max) as reference.
		log_S = librosa.power_to_db(S, ref=np.max)
		log_S = log_S.reshape(log_S.shape[0],-1,1) if reshape else log_S

		return 	log_S


def mfcc(samples,sample_rate=16000,reshape=True):
	log_S = log_mel(samples,reshape=False)
	mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=40)
	# delta2_mfcc = librosa.feature.delta(mfcc, order=2)
	del log_S
	mfcc = mfcc.reshape(mfcc.shape[0],-1,1) if reshape else mfcc

	return 	mfcc



def AS_to_raw(as_file):
	wav = np.array(as_file.get_array_of_samples().tolist())      
#     wav = wav.astype(np.float32) / np.iinfo(np.int16).max
	return wav



def fill_to_1sec(wav):
    #fill to 1 second
    L = 1000 #16000  # 1 sec
    sample_rate = 16000
    
    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        wav = wav[i:(i+L)]
    elif len(wav) < L:
        rem_len = L - len(wav)
        wav = AudioSegment.silent(rem_len,frame_rate=sample_rate) + wav
        
    return wav    







# def augment_wav(wav,pval=0.5):
#     sample_rate = 16000
#     L = 1000 #16000  # 1 sec
    
#     #adjust speed, with 50% chance
# #     wav = speed_change(wav,1.+random.uniform(-1, 1)*0.05) if np.random.random() < pval else wav
    
    
#     #adjust volume
#     db_adjustment = random.uniform(-1, 1)*10
#     wav = wav + db_adjustment if np.random.random() < pval else wav
     
        
#     #fill to 1 second
#     wav = fill_to_1sec(wav)        
        
#     #shift the audio by 10 ms
#     shift_length = 100
#     if np.random.random() < 0.5: #shift to left
#         wav = wav[:L-shift_length]+ AudioSegment.silent(shift_length,frame_rate=sample_rate)
#     else: #shift to right
#         wav = AudioSegment.silent(shift_length,frame_rate=sample_rate) + wav[shift_length:]
        
        
        
#     #blend original file with background noise     
#     if np.random.random() < pval:
#         noise = random.choice(silence_files_AS)
#         db_delta = (wav.dBFS - noise.dBFS) -10.

#         if db_delta< 0: #reduce intensity of loud background; if it's too silent, leave it be
#             noise = noise  + db_delta
#         wav = wav.overlay(noise)
 
#     return wav
