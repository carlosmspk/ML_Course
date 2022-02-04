from numpy import inner
import pandas as pd

res_df = pd.read_csv("DeepLearning/results.csv", index_col="id")
print (res_df.head())