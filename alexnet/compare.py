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

parser = argparse.ArgumentParser(description='Style loss')
parser.add_argument('son1', metavar='son1', type=str, help='filename')
parser.add_argument('son2', metavar='son2', type=str, help='filename')

args = parser.parse_args()
arg1 = args.son1
arg2 = args.son2

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


y1, sr1 = librosa.load(arg1)
y2, sr2 = librosa.load(arg2)
y1_array = np.asarray(y1)
y2_array = np.asarray(y2)
y1_reshaped = numpy.reshape(y1_array, (50, int(y1_array.shape[0]/50)))
y2_reshaped = numpy.reshape(y2_array, (50, int(y2_array.shape[0]/50)))
print(np.array_equal(y1_reshaped, y2_reshaped))
c1 = np.empty((1, 10, 50, int(y1_array.shape[0]/50)))
c1[0] = y1_reshaped
c2 = np.empty((1, 10, 50, int(y2_array.shape[0]/50)))
c2[0] = y2_reshaped
print('loading model')
model = load_model('models/model.h5')
print(np.array_equal(c1,c2))


def compare(c1, c2):
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_4').output)
        output_style = intermediate_layer_model.predict(c1)
        output_target = intermediate_layer_model.predict(c2)
        channels = output_style.shape[1]
        width = output_style.shape[2]
        height = output_style.shape[3]
        s_loss = style_loss(style_matrix(output_style[0]), style_matrix(output_target[0]), width, height, channels)
        return s_loss

print(compare(c1,c2))
