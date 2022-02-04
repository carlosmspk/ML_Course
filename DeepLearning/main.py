import pandas as pd

# store dataset as Data Frame (table format)
df = pd.read_csv("DeepLearning/dataset/WineQT.csv", index_col="Id")
print ("\n>>> First 5 rows of data:\n\n",df.head())


corr_dict = {
    "feature": [],
    "correlation": []
}
for corr, key in zip(df.corr().values[:,-1], df):
    if key != "quality":
        corr_dict["feature"].append(key)
        corr_dict["correlation"].append(abs(corr))

corr_df = pd.DataFrame(corr_dict)
corr_df = corr_df.sort_values("correlation", ascending=False)
print ("\n>>> Correlation between wine quality and measured features: \n\n", corr_df)

#### Build the Model

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

x = df.values[:,:-1]
y = df.values[:,-1]

print (x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .7, random_state=0)
