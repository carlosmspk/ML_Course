from keras.datasets import mnist
from utils import preprocess_data, create_model, fit_and_train, plot_results, load
from keras.models import load_model
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if __name__ == "__main__":

    METRICS_PATH="DeepLearning/results/mnist_history.pkl"
    MODEL_PATH="DeepLearning/results/mnist_last_model"

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test, shrink=.2)

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