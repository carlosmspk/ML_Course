from matplotlib import image as img
import numpy as np
from os import listdir, chdir

LOAD_SMALL_SUBSET = True
N_IMGS_TO_LOAD = 3

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
y = np.zeros((n_images, n_labels))

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
        y[i][label] = 1
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

def create_model(input_shape, output_labels) -> Sequential:
    # Stack NN layers
    model = Sequential()
    model.add(Flatten(input_shape=(input_shape)))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(output_labels))
    
    # create loss function
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam",
    loss=loss_fn,
    metrics=["accurcay"])

    return model

model = create_model((300,300, 3), y.shape[1])