from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.utils import plot_model
import graphviz
import pydot
import numpy
import librosa
import csv
from tqdm import tqdm
from keras import backend as K

numpy.set_printoptions(threshold=numpy.nan)

data = json.load(open('../nsynth-test/examples.json'))

y_array = []
sr_array = []
train_labels = []
cpt = 0
batch_size = 100 #len(data.keys())
with tqdm(total=batch_size) as pbar:
	for cle in data.keys():
		#print(cle)
		y, sr = librosa.load('../nsynth-test/audio/'+ cle +'.wav')
		y_array.append(y)
		sr_array.append(sr)
		train_labels.append(data[cle]['instrument_family'])
		#print(data[cle]['instrument_family'])
		pbar.update(1)
		cpt = cpt + 1
		if cpt >= batch_size:
			break

X_train = np.asarray(y_array)
X_train_new = np.memmap('features.myarray', dtype=np.float64, mode='w+', shape=(X_train.shape[0], 10, 50, int(X_train.shape[1]/50)))

with tqdm(total=X_train.shape[0]*10) as pbar:
	for i in range(X_train.shape[0]):
		for j in range(10):
			X_train_new[i][j] = numpy.reshape(X_train[i], (50, int(X_train.shape[1]/50)))
			pbar.update(1)

X_train = X_train_new

Y_train = np.asarray(train_labels)
Y_train = np_utils.to_categorical(Y_train, 11)

model = load_model('model.h5')

model.summary()

plot_model(model, to_file='model.png')

activations = []

intermediate_layer_model = Model(inputs=model.input,
	outputs=model.get_layer('conv2d_1').output)

intermediate_output = intermediate_layer_model.predict(X_train)

activations.append(intermediate_output)

intermediate_layer_model = Model(inputs=model.input,
	outputs=model.get_layer('conv2d_2').output)

intermediate_output = intermediate_layer_model.predict(X_train)

activations.append(intermediate_output)

intermediate_layer_model = Model(inputs=model.input,
	outputs=model.get_layer('conv2d_3').output)

intermediate_output = intermediate_layer_model.predict(X_train)

activations.append(intermediate_output)

intermediate_layer_model = Model(inputs=model.input,
	outputs=model.get_layer('conv2d_4').output)

intermediate_output = intermediate_layer_model.predict(X_train)

activations.append(intermediate_output)

intermediate_layer_model = Model(inputs=model.input,
	outputs=model.get_layer('conv2d_5').output)

intermediate_output = intermediate_layer_model.predict(X_train)

activations.append(intermediate_output)

#print(activations)
for i in activations:
	print(i.shape)
#print(len(activations))

def gram_matrix(x, k, k_prime):
	a = x[k]
	a_prime = x[k_prime]
	G = np.empty((a.shape[0], a.shape[1]))

	for i in range(0, a.shape[0]):
		for j in range(0, a.shape[1]):
			G[i][j] = np.dot(a[i][j], np.transpose(a_prime[i][j]))
	return G

print(gram_matrix(activations[0], 0, 1))

def style_matrix(x):
	gram = []
	for i in range(0, len(x)):
		for j in range(0, len(x)):
			gram.append(gram_matrix(x, i, j))
	return gram

def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))



classe = model.predict(X_train)

classeid = []
for i in range(0, classe.shape[0]):
    max = classe[i][1]
    maxid = 1
    for j in range(1, classe.shape[1]):
        if max < classe[i][j]:
            max = classe[i][j]
            maxid = j
    classeid.append(maxid+1)


#for i in range(0,batch_size):
#	print(str(classeid[i]) + ' vs ' + str(train_labels[i]+1))

correct = 0
for i in range(0, batch_size):
	if(classeid[i] == train_labels[i]+1):
		correct = correct + 1

print(str(correct/batch_size*100) + '%')
