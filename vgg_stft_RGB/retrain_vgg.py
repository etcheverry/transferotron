from keras.applications import vgg19
from vgg19_sound import VGG19_SOUND
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import scipy
import numpy as np
import json
import librosa
import cmath
from tqdm import tqdm
import random
import colorsys


def norm_to_rgb(value):
        r, g, b = colorsys.hsv_to_rgb(value, 0.7, 0.7)
        return int(r*255), int(g*255), int(b*255)

def preprocess_image(x, cle):
	width, height = x.shape
	height = height - int(height/4)
	tmp = np.empty((width, height, 3))
	for i in range(width):
		for j in range(height):
			pol = cmath.polar(x[i][j])
			tmp[i][j][0] = pol[0]
	maximum = np.max(np.abs(tmp))
	for i in range(width):
		for j in range(height):
                        val = tmp[i][j][0]
                        r,g,b = norm_to_rgb(tmp[i][j][0]/maximum)
                        tmp[i][j][0] = r
                        tmp[i][j][1] = g
                        tmp[i][j][2] = b
	#scipy.misc.imsave('images/' + cle + '.png', tmp)
	ret = np.expand_dims(tmp, axis=0) 
	ret = vgg19.preprocess_input(ret)
	return ret

data = json.load(open('nsynth-test/examples.json'))

y_array = []
sr_array = []
train_labels = []
cpt = 0
batch_size = 2000#len(data.keys())
with tqdm(total=batch_size) as pbar:
	#for cle in data.keys():
        while(cpt < batch_size):
                
		#print(cle)
                y, sr = librosa.load('./nsynth-test/audio/'+ cle +'.wav')
                y_stft = librosa.stft(y, n_fft=700)
                y_array.append(preprocess_image(y_stft, cle)[0])
                sr_array.append(sr)
                train_labels.append(data[cle]['instrument_family'])
		#print(data[cle]['instrument_family'])
                pbar.update(1)
                cpt = cpt + 1
                if cpt >= batch_size:
                        break

X_train = np.asarray(y_array)
print(X_train.shape)

Y_train = np.asarray(train_labels)
Y_train = np_utils.to_categorical(Y_train, 11)

print('X train :')
print(X_train.shape)
print(X_train[0].shape)


#print(X_train)
print('Y train :')
print(Y_train.shape)
print(Y_train)


model = VGG19_SOUND(input_shape=X_train[0].shape, weights=None, include_top=True, classes=11)

#model.summary()
sgd = optimizers.SGD(lr=1e-7, decay=1e-10, momentum=0.2, clipvalue=0.5)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=["accuracy"])

checkpointer = ModelCheckpoint("./models/vgg_sound4.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
model.fit(X_train, Y_train, validation_split=0.3, shuffle=True, epochs=30, batch_size=24, callbacks=[checkpointer])

