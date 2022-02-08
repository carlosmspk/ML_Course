
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def preprocess_data(x_train, y_train, x_test, y_test):
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

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
    
    model.summary()

    return model