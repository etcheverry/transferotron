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
from tqdm import tqdm
from keras import backend as K
from scipy.optimize import minimize, rosen, rosen_der
from scipy import optimize
from pyswarm import pso

print(len(sys.argv))
if(len(sys.argv)!=4):
	print("Usage: python3 predict.py style.wav content.wav output_name.wav")
	sys.exit(-1)

#numpy.set_printoptions(threshold=numpy.nan)

y_array = []
sr_array = []
y, sr = librosa.load(sys.argv[1])
y_array.append(y)
sr_array.append(sr)

X_style = np.asarray(y_array)
X_style_new = np.empty((X_style.shape[0], 10, 50, int(X_style.shape[1]/50)))

with tqdm(total=X_style.shape[0]*10) as pbar:
	for i in range(X_style.shape[0]):
		for j in range(10):
			X_style_new[i][j] = numpy.reshape(X_style[i], (50, int(X_style.shape[1]/50)))
			pbar.update(1)

X_style = X_style_new

y_array = []
sr_array = []
y, sr = librosa.load(sys.argv[2])
y_array.append(y)
sr_array.append(sr)


X_base = np.asarray(y_array)

X_target = np.asarray(y_array)

X_base_new = np.empty((X_base.shape[0], 10, 50, int(X_base.shape[1]/50)))

with tqdm(total=X_base.shape[0]*10) as pbar:
	for i in range(X_base.shape[0]):
		for j in range(10):
			X_base_new[i][j] = numpy.reshape(X_base[i], (50, int(X_base.shape[1]/50)))
			pbar.update(1)

X_base = X_base_new

model = load_model('model.h5')

#model.summary()

#plot_model(model, to_file='model.png')

activations_style = []
activations_target = []

#print(activations)

for i in activations_style:
	print(i.shape)
print('----')
for i in activations_target:
	print(i.shape)
#print(len(activations))

def gram_matrix(x, k, k_prime):
	a = x[0][k]
	a_prime = x[0][k_prime]
	G = 0
	for i in range(0, a.shape[0]):
		for j in range(0, a.shape[1]):
			G = G + a[i][j]*a_prime[i][j]
	return G

#print(gram_matrix(activations_style[0], 0, 1))

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


#print('Style loss :')
#print(style_loss(style_matrix(activations_style[0]), style_matrix(activations_target[0]), width, height, channels))

def content_loss(base, combination):
	b = base[0]
	c = combination[0]
	loss = 0
	for k in range(0, base.shape[0]):
		for i in range(0, base.shape[1]):
			for j in range(0, base.shape[2]):
				loss = loss + (b[i][j][k] - c[i][j][k])*(b[i][j][k] - c[i][j][k])
	return loss/(2*base.shape[0]*base.shape[1]*base.shape[2])

#print('Content loss :')
#print(content_loss(activations_style[0], activations_target[0]))
intermediate_layer_model = Model(inputs=model.input,
	outputs=model.get_layer('conv2d_1').output)
output_style = intermediate_layer_model.predict(X_style)
output_target = intermediate_layer_model.predict(X_base)
channels = output_style.shape[1]
width = output_style.shape[2]
height = output_style.shape[3]
loss = style_loss(style_matrix(output_style), style_matrix(output_target), width, height, channels)
grads = K.gradients(loss, X_base)

iteration = 0

def predict_sound(target):
    global iteration
    it = iteration%100
    if it == 0:
        print(target.shape)
        librosa.output.write_wav('sounds_pso/' + str(iteration/100) + sys.argv[3], target, sr, norm=False)
    iteration = iteration + 1
    
    target = np.reshape(target, (1, target.shape[0]))
    target_new = np.empty((target.shape[0], 10, 50, int(target.shape[1]/50)))
    print(target)
    for i in range(target.shape[0]):
        for j in range(10):
            target_new[i][j] = numpy.reshape(target[i], (50, int(target.shape[1]/50)))

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)
    output_style = intermediate_layer_model.predict(X_style)
    output_base = intermediate_layer_model.predict(X_base)
    output_target = intermediate_layer_model.predict(target_new)
    channels = output_style.shape[1]
    width = output_style.shape[2]
    height = output_style.shape[3]

    s_loss = style_loss(style_matrix(output_style), style_matrix(output_target), width, height, channels)

    print(s_loss)
    return s_loss


print(X_target.shape)
lb = np.empty(X_target.shape[1])
ub = np.empty(X_target.shape[1])

for i in range(lb.shape[0]):
    lb[i]=-1
    ub[i]=1

out, fout = pso(predict_sound, lb, ub)


