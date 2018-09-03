import librosa
import argparse
import pysndfile.sndio
import numpy as np
import cmath
import math



parser = argparse.ArgumentParser(description='Intensity curve transfer')
parser.add_argument('sound', metavar='s1', type=str,
                    help='Path to the sound.')
parser.add_argument('out', metavar='out', type=str,
                    help='Path to the output sound.')
parser.add_argument('sine', nargs = '*', type=str,
                    help='nb of sines.')

def generate_sinus(A, f, ph, dt):
    size = dt*fech
    s = np.empty((size))
    for i in range(size):
        s[i] = A * math.sin(2*math.pi*f*(i/fech) + ph)
    return s

args = parser.parse_args()
path = args.sound
sine = args.sine
print(sine)
out_path = args.out

snd, sr = librosa.load(path, sr=None)
A = 0.7
ph = 0
fech = 16000
for i in range(len(sine)):
    for j in range(len(snd)):
        snd[j] = snd[j] + A * math.sin(2*math.pi*int(sine[i])*(j/fech) + ph)
for j in range(len(snd)):
        snd[j] = snd[j]/(len(sine)+1)

pysndfile.sndio.write(out_path, snd, rate=sr, format='wav', enc='pcm16')

