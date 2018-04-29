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

def gram_matrix(x, k, k_prime):
	a = x[0][k]
	a_prime = x[0][k_prime]
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

data = json.load(open('nsynth-valid/examples.json'))
y_array = []
sr_array = []
train_labels = []
cpt = 0
batch_size = len(data.keys())
with tqdm(total=batch_size) as pbar:
	for cle in data.keys():
		y, sr = librosa.load('./nsynth-valid/audio/'+ cle +'.wav')
		y_array.append(y)
		sr_array.append(sr)
		pbar.update(1)
		cpt = cpt+1
		if cpt >= batch_size:
			break

X_train = np.asarray(y_array)
X_train_new = np.memmap('x_train.myarray', dtype=np.float64, mode='w+', shape=(X_train.shape[0], 10, 50, int(X_train.shape[1]/50)))

with tqdm(total=X_train.shape[0]*10) as pbar:
	for i in range(X_train.shape[0]):
		for j in range(10):
			X_train_new[i][j] = numpy.reshape(X_train[i], (50, int(X_train.shape[1]/50)))
			pbar.update(1)

X_train = X_train_new


model = load_model('model.h5')

print(X_train.shape)

output = []

cpt = 0
with tqdm(total=batch_size) as pbar:
	for cle in data.keys():
		intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_4').output)
		tmp = X_train[cpt].reshape((1, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
		output.append(intermediate_layer_model.predict(tmp))
		pbar.update(1)
		cpt = cpt+1
		if cpt >= batch_size:
			break

s_loss = []
d = np.memmap('losses.myarray', dtype=np.float64, mode='w+', shape=(len(output), len(output)))

with tqdm(total=len(output)*(len(output)+1)/2) as pbar:
	for i in range(0, len(output)):
		for j in range(i, len(output)):
			channels = output[i].shape[1]
			width = output[i].shape[2]
			height = output[i].shape[3]
			s_loss= style_loss(style_matrix(output[i]), style_matrix(output[j]), width, height, channels)
			d[i][j] = s_loss
			d[j][i] = s_loss
			pbar.update(1)

print(d)

numpy.save('losses', d)


def compare(c1, c2):

	intermediate_layer_model = Model(inputs=model.input,
		outputs=model.get_layer('conv2d_4').output)
	output_style = intermediate_layer_model.predict(c1)
	output_target = intermediate_layer_model.predict(c2)
	channels = output_style.shape[1]
	width = output_style.shape[2]
	height = output_style.shape[3]
	s_loss = style_loss(style_matrix(output_style), style_matrix(output_target), width, height, channels)
	return s_loss