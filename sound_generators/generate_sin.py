import librosa
import numpy as np
import math

fech = 16000

def generate_sinus(A, f, ph, dt):
    size = dt*fech
    s = np.empty((size))
    for i in range(size):
        s[i] = A * math.sin(2*math.pi*f*(i/fech) + ph)
    return s

s = generate_sinus(1, 220, 0, 4)

librosa.output.write_wav('sine220.wav', s, fech)
