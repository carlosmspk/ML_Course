from pickle import dump
from pickle import load as ld
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

def to_categorical_int(target:np.ndarray):
    result = np.zeros((target.shape[0], 10))
    result[np.arange(target.shape[0]),target] = 1
    return result
    
def preprocess_data(x_train, y_train, x_test, y_test, shrink: float = 1.0):
    if shrink < 1.0:
        train_slice = int(shrink*x_train.shape[0])
        test_slice = int(shrink*x_test.shape[0])

        x_train = x_train[:train_slice]
        y_train = y_train[:train_slice]
        x_test = x_test[:test_slice]
        y_test = y_test[:test_slice]

    y_train, y_test = to_categorical_int(y_train), to_categorical_int(y_test)

    # Add grayscale channel
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    # Convert to floats (for next step)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # Convert from integer 0-255 value to float 0-1 value
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test

def create_model(input_shape, label_num) -> Sequential:
    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu',input_shape=input_shape))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(32, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(label_num, activation = "softmax"))
    
    model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

    return model

def fit_and_train (model : Sequential, x_train, y_train, x_test, y_test, epochs = 1, validation_split = 0.2, store_metrics = None):
    accuracy = {
        "train" : [0],
        "val" : [0],
        "test" : [0]
    }
    epochs = [i+1 for i in range(epochs)]
    for epoch in epochs:
        print (f"Epoch: {epoch}/{len(epochs)}")
        fit_result = model.fit(x=x_train, y=y_train, epochs=1, validation_split=validation_split).history
        test_result = model.evaluate(x=x_test, y=y_test)

        accuracy["train"].append(fit_result["accuracy"][-1])
        accuracy["val"].append(fit_result["val_accuracy"][-1])
        accuracy["test"].append(test_result[1])

    if store_metrics is not None:
        with open(store_metrics, "wb") as f:
            dump(accuracy, f)
    return accuracy

def plot_results (accuracy_dict):
    acc_train = accuracy_dict["train"]
    acc_val = accuracy_dict["val"]
    acc_test = accuracy_dict["test"]
    assert len(acc_train) == len(acc_val) == len(acc_test)
    epochs = [i+1 for i in range (len(acc_train))]
    perfect_acc_y = [1,1]
    perfect_acc_x = [min(epochs),max(epochs)]

    plt.plot (epochs, acc_train, label = "Train accuracy")
    plt.plot (epochs, acc_val, label = "Validation accuracy")
    plt.plot (epochs, acc_test, label = "Test accuracy")
    plt.plot (perfect_acc_x, perfect_acc_y, "b--", label = "Perfect accuracy")
    plt.legend ()
    plt.show()

def load(path : str):
    with open(path, "rb") as f:
        loaded_data = ld(f)
    return loaded_data