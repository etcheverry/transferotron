import librosa
import argparse
import pysndfile.sndio
import numpy as np
import cmath

parser = argparse.ArgumentParser(description='Intensity curve transfer')
parser.add_argument('sound1', metavar='s1', type=str,
                    help='Path to the sound 1.')
parser.add_argument('sound2', metavar='s2', type=str,
                    help='Path to the sound 2.')
parser.add_argument('out', metavar='out', type=str,
                    help='Path to the output sound.')

#faire par frame

args = parser.parse_args()
s1_path = args.sound1
s2_path = args.sound2
out_path = args.out

s1, sr = librosa.load(s1_path, sr=None)
s2, sr = librosa.load(s2_path, sr=None)

frame_size = 20

for frame in range(int(len(s2)/frame_size)):
    moy = 0
    maximum = 0
    for i in range(frame, frame + frame_size):
        if(i < len(s2)):
            if(abs(s1[i]) > maximum):
                maximum = abs(s1[i])
            moy = moy + abs(s1[i])
    moy = moy/frame_size
    print(str(moy) + ' / ' + str(maximum) + ' = ' + str(moy/maximum))
    for i in range(frame, frame + frame_size):
        if(i < len(s2)):
            s2[i] = s2[i]*moy

            
'''
stft1 = librosa.stft(s1)
stft2 = librosa.stft(s2)

#Normalize stft1

width, height = stft1.shape
max_i = 0
for i in range(width):
    for j in range(height):
        if(cmath.polar(stft1[i][j])[0] > max_i):
            max_i = cmath.polar(stft1[i][j])[0]

curve = np.empty((width, height))
for i in range(width):
    for j in range(height):
        pol1 = cmath.polar(stft1[i][j])
        pol2 = cmath.polar(stft2[i][j])
        stft2[i][j] = cmath.rect(pol2[0]*(pol1[0]/max_i), pol2[1])

out = librosa.istft(stft2)'''
pysndfile.sndio.write(out_path, s2, rate=sr, format='wav', enc='pcm16')
