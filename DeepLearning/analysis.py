import pandas as pd
import pickle

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

plot_relationships(res_df)

#remove outlier
res_df_filtered = res_df[res_df['mean_test_score'] > -3]

plot_relationships(res_df_filtered)

def plot_predictions(filename):
    with open("DeepLearning/results/"+filename+".pickle", "rb") as f:
        data = pickle.load(f)

    x_test, y_test, model = data["x_test"], data["y_test"], data["model"]
    predictions = model.predict(x_test)

    plt.plot ((-2.8, 2.8), (-2.8,2.8), "r--")
    plt.scatter(predictions, y_test)
    plt.show()

plot_predictions("data_lessfeatures")