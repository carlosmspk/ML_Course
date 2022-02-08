from keras.datasets import mnist
from utils import preprocess_data, create_model, fit_and_train, plot_results, load
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if __name__ == "__main__":

    METRICS_PATH="DeepLearning/results/mnist_history.pkl"
    MODEL_PATH="DeepLearning/results/mnist_last_model"

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_original = x_test
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    try:
        model = load_model(MODEL_PATH)
        accuracy = load(METRICS_PATH)
        print("Found previously trained model and accuracy data.")
    except (FileNotFoundError, OSError):
        print("No stored model and/or accuracy values found. Creating new model...")
        model = create_model(x_train.shape[1:], 10)
        accuracy = fit_and_train (model, x_train, y_train, x_test, y_test, epochs=20, store_metrics=METRICS_PATH)
        model.save(MODEL_PATH)

    plot_results(accuracy)

    while True:
        i = np.random.randint(0, x_test.shape[0])
        x_img = x_original[i]
        x = x_test[i]
        plt.imshow(x_img, cmap=plt.get_cmap('gray'))
        single_point = np.array([x,])
        prediction = model.predict(single_point)[0]
        plt.title (np.argmax(prediction))
        plt.show()
        prompt = input("Show more? (Y/n)")
        if prompt == "n":
            break