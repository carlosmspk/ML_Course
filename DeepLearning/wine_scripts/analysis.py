import pandas as pd
import pickle

res_df = pd.read_csv("DeepLearning/results.csv", index_col="id")
print (res_df.head())

import matplotlib.pyplot as plt

def plot_relationships (df, target):
    def plot_in_axis(axis, column_label):
        axis.scatter(df[column_label], df[target])
        axis.set_title (column_label)

    _, axs = plt.subplots(1,3)
    for i, key in enumerate(df):
        if key == target:
            continue
        plot_in_axis(axs[i], key)
    plt.show()

plot_relationships(res_df, "mean_test_score")

#remove outlier
res_df_filtered = res_df[res_df['mean_test_score'] > -3]

plot_relationships(res_df_filtered, "mean_test_score")

def plot_predictions(filename):
    with open("DeepLearning/results/"+filename+".pickle", "rb") as f:
        data = pickle.load(f)

    x_test, y_test, model = data["x_test"], data["y_test"], data["model"]
    predictions = model.predict(x_test)

    start = min(min(y_test), min(predictions))
    stop = max(max(y_test), max(predictions))

    plt.plot ((start, stop), (start, stop), "r--")
    plt.scatter(predictions, y_test)
    plt.show()

plot_predictions("data_lessfeatures_class")

def plot_class_distances (filename):
    with open("DeepLearning/results/"+filename+".pickle", "rb") as f:
        data = pickle.load(f)

    x_test, y_test, model = data["x_test"], data["y_test"], data["model"]
    predictions = model.predict(x_test)

    distance_counts = dict()

    for y_true, y_predicted in zip(y_test, predictions):
        distance = abs(y_predicted-y_true)
        try:
            distance_counts[distance] += 1
        except KeyError:
            distance_counts[distance] = 1
    x = [key for key in distance_counts]
    y = [value for value in distance_counts.values()]

    plt.bar(x,y)
    plt.show()

plot_class_distances("data_lessfeatures_class")