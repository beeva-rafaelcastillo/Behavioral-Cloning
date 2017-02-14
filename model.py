"""
model to predict steering angles from the previous sequence of images
model architecture is taken from: https://github.com/jamesmf/mnistCRNN/blob/master/scripts/addMNISTrnn.py

References used:
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.a60sq6l6p
https://github.com/fchollet/keras/issues/1638
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

# Fix error with TF and Keras
import tensorflow as tf

import config
import process_data

tf.python.control_flow_ops = tf

from keras.models import Sequential

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping

def generator(samples, labels, batch_size):
    """
    Generator with augmented data to feed the model_RNN
    :param samples: numpy array with samples
    :param labels: numpy array with corresponding labels
    :param batch_size: int batch size
    :yields: batched samples augmented and corresponding labels
    """
    while 1:
        batch_images = []
        batch_steering = []
        for batch_sample in range(0, batch_size):
            # random value:
            intensity = np.random.uniform()
            # random flipping:
            flipping = np.random.choice([True, False])
            # random sample
            idx = np.random.randint(samples.shape[0])
            img_aug, steering_aug = process_data.augmented_images(samples[idx], labels[idx], flipping, intensity)
            batch_images.append(img_aug)
            batch_steering.append(steering_aug)
        batch_images = np.asarray(batch_images)
        batch_steering = np.asarray(batch_steering)
        yield batch_images, batch_steering


############################################################
# Configuration:
############################################################
input_files = '/home/carnd/Behavioral_Cloning/initial_files/'
current_path = '/home/carnd/Behavioral_Cloning/model_CNN/'
batch_size = 512
nb_epochs = 3
x_pix = config.x_pix
y_pix = config.y_pix
seed = 2016
test_size = 0.2
model_version = config.version

# for reproducibility
np.random.seed(seed)

############################################################
# Load and process Data_test:
############################################################
with open(input_files + 'features_{0}.pickle'.format(config.version), 'rb') as handle:
    X = pickle.load(handle)
with open(input_files + 'labels_{0}.pickle'.format(config.version), 'rb') as handle:
    y = pickle.load(handle)

X = X.astype('float32')
y = y.astype('float32')

# test:
# X = X[:700, :, :, :]
# y = y[:700]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)


############################################################
# Define our time-distributed setup
############################################################
model = Sequential()
model.add(Convolution2D(24, 4, 4, subsample=(2, 2), activation='relu', input_shape=(y_pix, x_pix, 3)))
model.add(Convolution2D(36, 4, 4, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 4, 4, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1))
# keras model compile, choose optimizer and loss func
model.compile(optimizer='adam', loss='mse')

# train generator:
train_generator = generator(X_train, y_train, batch_size=batch_size)
validation_generator = generator(X_test, y_test, batch_size=batch_size)

# callback:
early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

# run epochs of sampling data then training
model.fit_generator(train_generator, samples_per_epoch=batch_size*100, nb_epoch=nb_epochs, verbose=1,
                    validation_data=validation_generator, nb_val_samples=X_test.shape[0])

# evaluate:
print("Model Evaluation: ", model.evaluate(X_test, y_test, batch_size=32, verbose=0, sample_weight=None))

# save the model
model.save(current_path + 'model_{0}.h5'.format(model_version))
print("model Saved!")

print("Model structure:")
print(model.summary())
