from matplotlib import image as img
import numpy as np
from os import listdir, chdir

LOAD_SMALL_SUBSET = True
N_IMGS_TO_LOAD = 10

chdir("DeepLearning")
n_images = 0
n_labels = 0
encoder = dict()
for dirname in listdir("dataset/IndianFood"):
    if dirname == ".DS_Store":
        continue
    for image in listdir("dataset/IndianFood/" + dirname):
        n_images += 1
    encoder[n_labels] = dirname
    n_labels += 1

if LOAD_SMALL_SUBSET:
    n_images = N_IMGS_TO_LOAD * n_labels

label = 0
x = np.zeros((n_images,300,300,3))
y = np.zeros((n_images,))

i = 0
label = 0

for dirname in listdir("dataset/IndianFood"):
    if dirname == ".DS_Store":
        continue
    print (f"Loading {dirname}... ", end="", flush=True)
    loaded_images = 0
    for image in listdir("dataset/IndianFood/" + dirname):
        if LOAD_SMALL_SUBSET and loaded_images >= N_IMGS_TO_LOAD:
            break
        loaded_images += 1
        x[i] = img.imread("dataset/IndianFood/" + dirname + "/" + image)
        y[i] = label
        i += 1
    label += 1
    print("done!")

print (x.shape)
print (y.shape)


# Start building model

import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy

def create_model(input_sample, label_num) -> Sequential:
    # Stack NN layers
    model = Sequential()
    model.add(Flatten(input_shape=input_sample.shape))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(label_num))
    
    # create loss function
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam",
    loss=loss_fn,
    metrics=["accuracy"])

    return model

model = create_model(x[0], n_labels)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)