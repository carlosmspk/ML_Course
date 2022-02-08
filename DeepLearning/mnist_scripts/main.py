from keras.datasets import mnist
from utils import preprocess_data, create_model, fit_and_train, plot_results
from pickle import load


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)

    model = create_model(x_train.shape[1:], 10)
    accuracy = fit_and_train (model, x_train, y_train, x_test, y_test, epochs=20, store_metrics="DeepLearning/results/mnist_history.pkl")
    plot_results(accuracy)