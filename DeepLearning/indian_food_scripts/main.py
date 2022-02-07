import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from batch import BatchGenerator
import numpy as np

DATASET_PATH = "DeepLearning/dataset/IndianFood/"
TEST_SIZE = .2
VALIDATION_SIZE = .2
BATCH_SIZE = 60
EPOCHS = 5

batch_gen = BatchGenerator(DATASET_PATH, BATCH_SIZE)

i_shape, o_shape = batch_gen.img_shape, batch_gen.output_shape[1]

def create_model(input_sample, label_num) -> Sequential:
    # Stack NN layers
    model = Sequential()
    model.add(Flatten(input_shape=input_sample.shape))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(label_num, activation="softmax"))
    
    # create loss function
    model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

    model.summary(line_length=42)

    return model

model = create_model(np.zeros(i_shape), o_shape)

model.fit_generator(generator=)
model.evaluate(x_test, y_test, verbose=2)