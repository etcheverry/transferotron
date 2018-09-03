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
import cmath

print(len(sys.argv))
if(len(sys.argv)!=4):
	print("Usage: python3 predict.py style.wav content.wav output_name.wav")
	sys.exit(-1)

#numpy.set_printoptions(threshold=numpy.nan)

channels=10

y_array = []
sr_array = []
y_stft = []
y, sr = librosa.load(sys.argv[1])
y_array.append(y)
sr_array.append(sr)
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

X_style = X_style_new


y_array = []
sr_array = []
y_stft = []
y, sr = librosa.load(sys.argv[2])
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

X_base = X_base_new

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


model = load_model('models/model2.h5')

iteration = 0

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

def shaped_array_to_sound(array):
        ch = array.shape[1]/2
        ret = np.empty((1, array.shape[2], array.shape[3]), dtype=np.complex_)
        for i in range(array.shape[0]):
                for x in range(array.shape[2]):
                        for y in range(array.shape[3]):
                                r = 0
                                phi = 0
                                for j in range(array.shape[1]):
                                        if(j < array.shape[1]/2):
                                                r = r + array[i][j][x][y]
                                        else:
                                                phi = phi + array[i][j][x][y]
                                ret[i][x][y] = cmath.rect(r/ch, phi/ch)
        return ret

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)
output_style = intermediate_layer_model.predict(X_style)
output_base = intermediate_layer_model.predict(X_base)

shape1 = X_base.shape[2]
shape2 = X_base.shape[3]
channels = output_style.shape[1]
        

loss = style_loss(style_matrix(output_style), style_matrix(output_base), shape1, shape2, channels)

grads = K.gradients(loss, output_base)

def predict_sound(target):
    
    global iteration
    it = iteration%100
    print(iteration)
    target_new = reshape1dTo3d(target, 10, shape1, shape2)
    if it == 0:
        print('writing sound')
        spectro = shaped_array_to_sound(target_new)
        sound = librosa.istft(spectro[0])
        librosa.output.write_wav('sounds_minimize/' + str(iteration/100) + sys.argv[3], sound, sr, norm=False)
        
    iteration = iteration + 1
    
    target_new = reshape1dTo3d(target, 10, shape1, shape2)

    output_base = intermediate_layer_model.predict(target_new)

    grad_values = np.array(output_base).flatten().astype('float64')


    
    channels = output_style.shape[1]
    width = output_style.shape[2]
    height = output_style.shape[3]

    s_loss = style_loss(style_matrix(output_style), style_matrix(output_base), width, height, channels)

    print(s_loss)

    print(grad_values)
    return s_loss, grad_values


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = predict_sound(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

X_base = reshape3dTo1d(X_base)



out, fout = optimize.fmin_l_bfgs_b(evaluator.loss, X_base, fprime=evaluator.grads, maxfun=20)
