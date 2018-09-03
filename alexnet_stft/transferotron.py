from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy
import json
import librosa
from tqdm import tqdm

from pprint import pprint
import cmath

data = json.load(open('../nsynth-test/examples.json'))

y_array = []
sr_array = []
train_labels = []
cpt = 0
batch_size = 1024#len(data.keys())

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

print('X train :')
print(X_train.shape)

#print(X_train)
print('Y train :')
print(Y_train.shape)
print(Y_train)


print('start training')
#keras


model = Sequential()
model.add(Conv2D(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), data_format='channels_first', filters=10, kernel_size=(1,1), strides=(2,2), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_first', strides=(2,2), padding='valid'))

'''
model.add(Conv2D(filters=10, data_format='channels_first', kernel_size=(10,10), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_first', strides=(2,2), padding='valid'))

model.add(Conv2D(filters=10, data_format='channels_first', kernel_size=(10,10), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_first', strides=(2,2), padding='valid'))

model.add(Conv2D(filters=10, data_format='channels_first', kernel_size=(10,10), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_first', strides=(2,2), padding='valid'))

model.add(Conv2D(filters=10, data_format='channels_first', kernel_size=(10,10), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1,1), data_format='channels_first', strides=(2,2), padding='valid'))
'''
model.add(Flatten())

model.add(Dense(4096))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(11))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=["accuracy"])

checkpointer = ModelCheckpoint("./models/model.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.fit(X_train, Y_train, validation_split=0.3, epochs=10, batch_size=20, callbacks=[checkpointer])

model.summary()

model.save('model.h5')
