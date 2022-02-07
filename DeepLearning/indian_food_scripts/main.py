from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from batch import BatchGenerator
from pickle import load

DATASET_PATH = "DeepLearning/dataset/IndianFood/"
TEST_SIZE = .2
VALIDATION_SIZE = .2
BATCH_SIZE = 100
EPOCHS = 10

# Fetch image paths and labels
with open("DeepLearning/dataset/IndianFood/data.pkl", "rb") as f:
    data = load(f)
img_paths = data["img_paths"]
labels = data["labels"]

x_train, x_test, y_train, y_test = train_test_split(img_paths, labels, test_size=TEST_SIZE)
x_train, x_val, y_train, y_val = train_test_split(img_paths, labels, test_size=VALIDATION_SIZE)

batch_gen_train = BatchGenerator(dataset_path=DATASET_PATH, image_paths=x_train, labels=y_train, batch_size=BATCH_SIZE)
batch_gen_val = BatchGenerator(dataset_path=DATASET_PATH, image_paths=x_val, labels=y_val, batch_size=BATCH_SIZE)
batch_gen_test = BatchGenerator(dataset_path=DATASET_PATH, image_paths=x_test, labels=y_test, batch_size=BATCH_SIZE)

def create_model(input_shape, label_num) -> Sequential:
    # Stack NN layers
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(label_num, activation="softmax"))
    
    # create loss function
    model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

    print(model.summary(line_length=42))

    return model

model = create_model((300,300,3), 9)

model.fit_generator(generator=batch_gen_train,
                   steps_per_epoch = int(y_train.shape[0] // BATCH_SIZE),
                   epochs = EPOCHS,
                   verbose = 1,
                   validation_data = batch_gen_val,
                   validation_steps = int(y_val.shape[0] // BATCH_SIZE))
result = model.evaluate_generator(generator=batch_gen_test, verbose=1)