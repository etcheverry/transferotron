import librosa
import numpy as np
import pysndfile.sndio

fech = 16000
dt = 4
size = dt*fech
s = np.empty((size))

pysndfile.sndio.write('silence.wav', s, rate=fech, format='wav', enc='pcm16')
