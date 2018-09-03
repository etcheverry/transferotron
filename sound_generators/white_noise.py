import librosa
import numpy as np
import math
import random
import cmath
import pysndfile.sndio

fech = 16000

def generate_white_noise(dt):
    size = dt*fech
    s = np.empty((size))
    fft = librosa.stft(s)
    for i in range(fft.shape[0]):
        for j in range(fft.shape[1]):
            fft[i][j] = cmath.rect(10, random.uniform(0,10))
    s = librosa.istft(fft)
    return s

s = generate_white_noise(4)


pysndfile.sndio.write('white_noise.wav', s, rate=fech, format='wav', enc='pcm16')
