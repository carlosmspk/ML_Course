from matplotlib import image as img
import numpy as np
from os import listdir, chdir

load_small_subset = True

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
        if load_small_subset and loaded_images >= 3:
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

def create_model(input_sample: np.ndarray, output_labels) -> Sequential:
    # Stack NN layers
    model = Sequential()
    model.add(Flatten(input_shape=(input_sample.shape)))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(output_labels))
    
    # create loss function
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam",
    loss=loss_fn,
    metrics=["accurcay"])

    return model

model = create_model(x[0].shape, y.shape[1])