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
import ntpath


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

def load_folder(name):
        y_array = []
        labels =[]
        for file in os.listdir(name):
                if file.endswith(".wav"):
                        labels.append(ntpath.basename(file))
                        y, sr = librosa.load(os.path.join(name, file))
                        y_array.append(y)
        y2_array = np.asarray(y_array)
        y2_array_new = np.empty((y2_array.shape[0], 10, 50, int(y2_array.shape[1]/50)))
        with tqdm(total=y2_array.shape[0]*10) as pbar:
                for i in range(y2_array.shape[0]):
                        for j in range(10):
                                y2_array_new[i][j] = numpy.reshape(y2_array[i], (50, int(y2_array.shape[1]/50)))
                                pbar.update(1)
        return y2_array_new, labels

def compare(file, folder):
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_4').output)
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

def load_file(name):
        y1, sr1 = librosa.load(name)
        y1_array = np.asarray(y1)
        y1_reshaped = numpy.reshape(y1_array, (50, int(y1_array.shape[0]/50)))
        c1 = np.empty((1, 10, 50, int(y1_array.shape[0]/50)))
        c1[0] = y1_reshaped
        return c1


# Variation de pitch
file1 = load_file('../outs/pitch_variation/guitar_acoustic_014-058-075.wav')
folder1, labels1 = load_folder('../outs/pitch_variation/')

# Variation d'instruments
file2 = load_file('../outs/instr_variation/guitar_acoustic_014-057-050.wav')
folder2, labels2 = load_folder('../outs/instr_variation')

# Bruit blanc
file_wn = load_file('../outs/white_noise/0_white_noise.wav')
folder_wn, labels_wn = load_folder('../outs/white_noise/')

# Silence
file_sil = load_file('../outs/silence/0_silence.wav')
folder_sil, labels_sil = load_folder('../outs/silence')


print('loading model')
model = load_model('models/model.h5')
x = []
cmp = compare(file1, folder1)
for i in range(len(labels1)):
        x.append(int(labels1[i][:-8][-3:]))
plt.scatter(x, cmp, marker='x')
plt.ylim(0, 2500000)

plt.savefig('pitch_variation.png')

plt.clf()
plt.cla()
plt.close()

cmp = compare(file2, folder2)
plt.scatter(range(len(cmp)),cmp, marker='x')
plt.ylim(0, 2500000)
for i in range(len(labels2)):
        labels2[i] = labels2[i][:-12]
        
plt.xticks(range(len(cmp)), labels2, rotation='vertical')
plt.subplots_adjust(bottom=0.40)
plt.savefig('instrument_variation.png')

plt.clf()
plt.cla()
plt.close()

cmp = compare(file_wn, folder_wn)
plt.scatter(range(len(cmp)),cmp, marker='x', color='tab:blue')

cmp = compare(file_sil, folder_sil)
plt.scatter(range(len(cmp)),cmp, marker='+', color='tab:red')
for i in range(len(labels_sil)):
        labels_sil[i] = labels_sil[i][:-12]
        
plt.xticks(range(len(cmp)), labels_wn, rotation='vertical')
plt.subplots_adjust(bottom=0.40)
plt.savefig('silence_wn.png')


