from numpy import inner
import pandas as pd

res_df = pd.read_csv("DeepLearning/results.csv", index_col="id")
print (res_df.head())

import matplotlib.pyplot as plt

def plot_relationships (df):
    def plot_in_axis(axis, column_label):
        axis.scatter(df[column_label], df["mean_test_score"])
        axis.set_title (column_label)

    _, axs = plt.subplots(1,3)
    for i, key in enumerate(df):
        if key == "mean_test_score":
            break
        plot_in_axis(axs[i], key)
    plt.show()
