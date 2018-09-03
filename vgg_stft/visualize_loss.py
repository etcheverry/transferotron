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
import ntpath

from keras.applications import vgg19

def preprocess_image(x):
    array = np.asarray(x)
    width, height = array.shape
    phases = np.empty((width, height))
    tmp = np.empty((width, height, 3))
    for i in range(width):
        for j in range(height):
            pol = cmath.polar(array[i][j])
            phases[i][j] = pol[1]
            for c in range(3):
                tmp[i][j][c] = pol[0]
    ret = np.expand_dims(tmp, axis=0)
    ret = vgg19.preprocess_input(ret)
    return ret, phases

def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination, img_nrows, img_ncols):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def load_file(name):
        y, sr = librosa.load(name, sr=None)
        stft = librosa.stft(y)
        return preprocess_image(stft)[0]

def load_folder(name):
        labels = []
        array = []
        for file in os.listdir(name):
                if file.endswith(".wav"):
                        labels.append(ntpath.basename(file))
                        array.append(load_file(os.path.join(name, file)))
        return array, labels

def compare(file, folder):
        ret = []
        file_var = K.variable(file)
        img_nrows, img_ncols = file.shape[1], file.shape[2]
        for f2 in folder:
                f2_var = K.variable(f2)

                # this will contain our generated image
                if K.image_data_format() == 'channels_first':
                    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
                else:
                    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

                # combine the 3 images into a single Keras tensor
                input_tensor = K.concatenate([file_var, f2_var, combination_image], axis=0)
                print('loading model')
                model = vgg19.VGG19(input_tensor=input_tensor, weights=None, include_top=False)
                model.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
                outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
                
                feature_layers = ['block1_conv1', 'block2_conv1',
                                  'block3_conv1', 'block4_conv1',
                                  'block5_conv1']
                sl = K.variable(0.)
                for layer_name in feature_layers:
                        layer_features = outputs_dict[layer_name]
                        style_reference_features = layer_features[1, :, :, :]
                        combination_features = layer_features[2, :, :, :]
                        sl += style_loss(style_reference_features, combination_features, img_nrows, img_ncols)/len(feature_layers)
                grads = K.gradients(sl, combination_image)

                outputs = [sl]
                if isinstance(grads, (list, tuple)):
                    outputs += grads
                else:
                    outputs.append(grads)
                f_outputs = K.function([combination_image], outputs)
                outs = f_outputs([file])
                loss_value = outs[0]
                ret.append(loss_value)

        return ret


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


x = []
cmp = compare(file1, folder1)
for i in range(len(labels1)):
        x.append(int(labels1[i][:-8][-3:]))
plt.scatter(x, cmp, marker='x')
plt.savefig('pitch_variation.png')

plt.clf()
plt.cla()
plt.close()

cmp = compare(file2, folder2)
plt.scatter(range(len(cmp)),cmp, marker='x')
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
        
plt.xticks(range(len(cmp)), labels_sil, rotation='vertical')
plt.subplots_adjust(bottom=0.40)
plt.savefig('silence_wn.png')

