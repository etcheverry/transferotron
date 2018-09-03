from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.utils import plot_model
import graphviz
import pydot
import numpy
import librosa
import csv
import sys
import pickle
from tqdm import tqdm
from keras import backend as K
from scipy.optimize import minimize, rosen, rosen_der
import argparse
import os
import matplotlib.pyplot as plt
import cmath

channels=10

def gram_matrix(x, k, k_prime):
        a = x[k]
        a_prime = x[k_prime]
        G = 0
        for i in range(0, a.shape[0]):
                for j in range(0, a.shape[1]):
                        G = G + a[i][j]*a_prime[i][j]
        return G

def style_matrix(x):
	channels = x.shape[1]
	gram = np.empty((channels,channels))
	for i in range(0, channels):
		for j in range(0, channels):
			gram[i][j] = gram_matrix(x, i, j)
	return gram

def style_loss(style, combination, width, height, channels):
    J = 0
    for i in range(0, style.shape[0]):
    	for j in range(0, style.shape[1]):
    		J = J + (style[i][j] - combination[i][j])*(style[i][j] - combination[i][j])
    return J/(2*width*height*channels)

def reshape3dTo1d(array):
        ret = np.empty((array.shape[1]*array.shape[2]*array.shape[3]))
        cpt = 0
        for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                        for x in range(array.shape[2]):
                                for y in range(array.shape[3]):
                                        ret[cpt] = array[i][j][x][y]
                                        cpt = cpt+1
        return ret

def reshape1dTo3d(array, shape1, shape2, shape3):
        ret = np.empty((1, shape1, shape2, shape3))
        cpt = 0
        for i in range(ret.shape[0]):
                for j in range(ret.shape[1]):
                        for x in range(ret.shape[2]):
                                for y in range(ret.shape[3]):
                                        ret[i][j][x][y] = array[cpt]
                                        cpt = cpt+1
        return ret

def load_folder(name):
        y_stft = []
        for file in os.listdir(name):
                if file.endswith(".wav"):
                        y, sr = librosa.load(os.path.join(name, file))
                        y_stft.append(librosa.stft(y))
                        
        X_style = np.asarray(y_stft)
        X_style_new = np.empty((X_style.shape[0], channels, X_style.shape[1], X_style.shape[2]))

        with tqdm(total=X_style.shape[0]*channels) as pbar:
                for i in range(X_style_new.shape[0]):
                        for j in range(X_style_new.shape[1]):
                                for x in range(X_style_new.shape[2]):
                                        for y in range(X_style_new.shape[3]):
                                                pol = cmath.polar(X_style[i][x][y])
                                                if j < X_style_new.shape[1]/2:
                                                        X_style_new[i][j][x][y] = pol[0]
                                                else:
                                                        X_style_new[i][j][x][y] = pol[1]
                                pbar.update(1)

        return X_style_new

def load_file(name):
        y_array = []
        sr_array = []
        y_stft = []
        y, sr = librosa.load(name)
        y_array.append(y)
        sr_array.append(sr)
        y_stft.append(librosa.stft(y))
                        
        X_base = np.asarray(y_stft)

        X_base_new = np.empty((1, 10, X_base.shape[1], X_base.shape[2]))
        for i in range(X_base_new.shape[0]):
                for j in range(X_base_new.shape[1]):
                        for x in range(X_base_new.shape[2]):
                                for y in range(X_base_new.shape[3]):
                                        pol = cmath.polar(X_base[i][x][y])
                                        if(j < X_base_new.shape[1]/2):
                                                X_base_new[i][j][x][y] = pol[0]
                                        else:
                                                X_base_new[i][j][x][y] = pol[1]

        return X_base_new

model = load_model('models/model2.h5')
        
def compare(file, folder):
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)
        output_style = intermediate_layer_model.predict(file)
        output_target = intermediate_layer_model.predict(folder)
        channels = output_style.shape[1]
        width = output_style.shape[2]
        height = output_style.shape[3]
        s_loss = []
        with tqdm(total=output_target.shape[0]) as pbar:
                for i in range(output_target.shape[0]):
                        s_loss.append(style_loss(style_matrix(output_style[0]), style_matrix(output_target[i]), width, height, channels))
                        pbar.update(1)
        return s_loss




# Variation de pitch
file1 = load_file('../outs/pitch_variation/guitar_acoustic_014-058-075.wav')
folder1 = load_folder('../outs/pitch_variation/')

# Variation d'instruments
file2 = load_file('../outs/instr_variation/guitar_acoustic_014-057-050.wav')
folder2 = load_folder('../outs/instr_variation')

# Bruit blanc
file_wn = load_file('../outs/white_noise/0_white_noise.wav')
folder_wn = load_folder('../outs/white_noise/')

# Silence
file_sil = load_file('../outs/silence/0_silence.wav')
folder_sil = load_folder('../outs/silence')


print('loading model')
model = load_model('models/model.h5')

plt.plot(compare(file1, folder1))
plt.savefig('pitch_variation.png')

plt.clf()
plt.cla()
plt.close()

plt.plot(compare(file2, folder2))
plt.savefig('instrument_variation.png')

plt.clf()
plt.cla()
plt.close()

plt.plot(compare(file_wn, folder_wn))
plt.plot(compare(file_sil, folder_sil))
plt.savefig('silence_wn.png')


