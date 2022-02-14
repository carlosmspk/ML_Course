from sklearn.model_selection import train_test_split
from keras.layers import (
    Flatten,
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)
from keras.models import Sequential, load_model
from batch import BatchGenerator
from pickle import load
from os import environ
import sys
from datetime import datetime
import numpy as np

f = open("out_log.txt", "a")
sys.stdout = f
print(
    "\n\n\n\n"
    + "=" * 65
    + "\n\tNEW LOG SESSION: "
    + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    + "\n"
    + "=" * 65
    + "\n\n"
)

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATASET_PATH = "DeepLearning/dataset/IndianFood/"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
BATCH_SIZE = 50
EPOCHS = 2

# Fetch image paths and labels
with open("DeepLearning/dataset/IndianFood/data.pkl", "rb") as f:
    data = load(f)
img_paths = data["img_paths"]
labels = data["labels"]

x_train, x_test, y_train, y_test = train_test_split(
    img_paths, labels, test_size=TEST_SIZE
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=VALIDATION_SIZE
)

batch_gen_train = BatchGenerator(
    dataset_path=DATASET_PATH,
    image_paths=x_train,
    labels=y_train,
    batch_size=BATCH_SIZE,
)
batch_gen_val = BatchGenerator(
    dataset_path=DATASET_PATH, image_paths=x_val, labels=y_val, batch_size=BATCH_SIZE
)
batch_gen_test = BatchGenerator(
    dataset_path=DATASET_PATH, image_paths=x_test, labels=y_test, batch_size=BATCH_SIZE
)


def create_model(input_shape, label_num) -> Sequential:
    model = Sequential()
    model.add(
        Conv2D(
            filters=64, kernel_size=(5, 5), activation="relu", input_shape=input_shape
        )
    )
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation="relu"))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=256, kernel_size=(5, 5), activation="relu"))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(60, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(label_num, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    return model


model = create_model((300, 300, 3), 9)

result_train = model.fit(
    x=batch_gen_train,
    steps_per_epoch=int(y_train.shape[0] // BATCH_SIZE),
    epochs=EPOCHS,
    verbose=0,
    validation_data=batch_gen_val,
)

result_train = result_train.history

accuracy_test = model.evaluate(batch_gen_test, verbose=0)[1]
accuracy_train, accuracy_val = (
    result_train["accuracy"][-1],
    result_train["val_accuracy"][-1],
)
print(
    f"Accuracy (train, val, test): {accuracy_train:.4f}, {accuracy_val:.4f}, {accuracy_test:.4f}"
)

model.save("DeepLearning/results/indian_last_model")

f.close()
