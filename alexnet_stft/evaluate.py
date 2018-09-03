from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.utils import plot_model
import graphviz
import pydot
import numpy
import librosa
import csv
import cmath
from tqdm import tqdm
from keras import backend as K

numpy.set_printoptions(threshold=numpy.nan)

data = json.load(open('../nsynth-test/examples.json'))

y_array = []
sr_array = []
train_labels = []
cpt = 0
batch_size = 10#len(data.keys())

channels = 10

y_stft = []

with tqdm(total=batch_size) as pbar:
	for cle in data.keys():
		#print(cle)
		y, sr = librosa.load('../nsynth-test/audio/'+ cle +'.wav')
		y_array.append(y)
		sr_array.append(sr)
		y_stft.append(librosa.stft(y))
		train_labels.append(data[cle]['instrument_family'])
		#print(data[cle]['instrument_family'])
		pbar.update(1)
		cpt = cpt +1
		if cpt >= batch_size:
			break



X_train = np.asarray(y_stft)

print('X_train shape : ')
print(X_train.shape)

X_train_new = np.memmap('features.myarray', dtype=np.float64, mode='w+', shape=(X_train.shape[0], channels, X_train.shape[1], X_train.shape[2]))

with tqdm(total=X_train.shape[0]*channels) as pbar:
	for i in range(X_train_new.shape[0]):
		for j in range(X_train_new.shape[1]):
                        for x in range(X_train_new.shape[2]):
                                for y in range(X_train_new.shape[3]):
                                        pol = cmath.polar(X_train[i][x][y])
                                        if j < X_train_new.shape[1]/2:
                                                X_train_new[i][j][x][y] = pol[0]
                                        else:
                                                X_train_new[i][j][x][y] = pol[1]
                        pbar.update(1)

X_train = X_train_new


Y_train = np.asarray(train_labels)
Y_train = np_utils.to_categorical(Y_train, 11)


model = load_model('models/model.h5')

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
