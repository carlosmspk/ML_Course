from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train, y_test = to_categorical(y_train), to_categorical(y_test)
print (x_train.shape, y_train.shape, x_test.shape, y_test.shape)

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

def fit_and_train (x_train, y_train, x_test, y_test):
    pass

model = create_model()
model.fit(x = x_train, y = y_train)
