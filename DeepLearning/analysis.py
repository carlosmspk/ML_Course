from numpy import inner
import pandas as pd

res_df = pd.read_csv("DeepLearning/results.csv", index_col="id")
print (res_df.head())

import matplotlib.pyplot as plt

plt.scatter(res_df["learning_rate"], res_df["mean_test_score"])
plt.xlabel("Learning Rate")
plt.ylabel("mean_test_score")
plt.show()