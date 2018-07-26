'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import random

#Import the Wandb libraries
import wandb
from wandb.keras import WandbCallback

#Set up the wandb environment
run = wandb.init()
config = run.config

#Track hyperparameters with wandb
config.num_classes = 10
config.batch_size = 64
config.epochs = 8
config.lr =  0.001
config.beta_1 = 0.9
config.beta_2 = 0.999


# Input image dimensions
img_rows, img_cols = 28, 28

# Split data between training and test
(x_train_orig, y_train_orig), (x_test, y_test) = mnist.load_data()

# Reducing the dataset size to 2/3rd of the original size for faster train time
true = list(map(lambda x: True if random.random() > 0.33 else False, range(60000)))
ind = []
for i, x in enumerate(true):
    if x == True: ind.append(i)

x_train = x_train_orig[ind, :, :]
y_train = y_train_orig[ind]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, config.num_classes)
y_test = keras.utils.to_categorical(y_test, config.num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(config.num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=config.beta_1, beta_2=config.beta_2),
              metrics=['accuracy'])

#Add the wandb callback to model.fit
model.fit(x_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[WandbCallback()])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])