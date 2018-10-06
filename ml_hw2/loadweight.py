import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
import os, sys
import numpy as np
import keras
import random
from skimage import io, transform, color
from tflearn.layers.conv import global_avg_pool
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.models import Model
import pandas as pd
from keras import layers
from keras import backend as K
import pathlib
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)

KTF.set_session(session)
PATH = pathlib.Path('CroppedYale').resolve()
SORTED_PATH = sorted([x for x in PATH.iterdir() if x.is_dir()])


def getdata():
    train_paths, train_lbls = [], []
    test_paths, test_lbls = [], []

    for i, j in enumerate(tqdm(SORTED_PATH)):
        paths = j.glob('*.pgm')
        paths = [p for p in sorted(paths)]
        label = [i for _ in range(len(paths))]
        
        train_paths.extend(paths[:35])
        train_lbls.extend(label[:35])
        test_paths.extend(paths[35:])
        test_lbls.extend(label[35:])

    xtrain = []
    for i, p in enumerate(tqdm(train_paths)):
        img = np.array(Image.open(p))
        if img.shape != (224, 224):
            img = transform.resize(img, (224, 224))
        colortrain = color.gray2rgb(img)
        assert colortrain.shape == (224, 224, 3)
        xtrain.append(colortrain)
    xtest = []
    for i, p in enumerate(tqdm(test_paths)):
        img = np.array(Image.open(p))
        if img.shape != (224,224):
                img = transform.resize(img, (224, 224))
        colortest = color.gray2rgb(img)
        assert colortest.shape == (224, 224, 3)
        xtest.append(colortest)

    return np.array(xtrain), np.uint8(train_lbls), np.array(xtest), np.uint8(test_lbls)



def VGG16(input_tensor=None, input_shape=None):

    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    '''
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fca')(x)
    x = Dense(4096, activation='relu', name='fcb')(x)
    x = Dense(12, activation='softmax', name='Classification')(x)
    '''
    # Create model.
    inputs = img_input
    model = Model(inputs,x,name='vgg16')
    return model

model = VGG16(input_shape=[224, 224, 3])
model.load_weights("model.h5")
output = model.output
output = Flatten()(output)
output = Dense(38, activation='softmax')(output)
model2 = Model(model.input, output)
model2.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])

model2.summary()


train_set, train_label, test_set, test_label = getdata()
print(train_set.shape)
print(train_label.shape)
print(test_set.shape)
print(test_label.shape)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model2.fit(train_set, train_label, batch_size=32,
                     epochs=30, validation_data=(test_set, test_label),callbacks=[early_stop],verbose=1)
exit()