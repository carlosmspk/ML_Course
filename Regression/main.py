import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats

show_plots = False

load_boston = load_boston()
x, y = load_boston.data, load_boston.target

data = pd.DataFrame(x, columns=load_boston.feature_names)
data["SalePrice"] = y
print(data.shape)
print(data.info())

if show_plots:
    sns.displot(data["SalePrice"])
    plt.show()
    stats.probplot(data["SalePrice"], plot=plt)
    plt.show()

corr = data.corr()

data["SalePrice"] = np.log1p(data["SalePrice"])
if show_plots:
    sns.displot(data["SalePrice"])
    plt.show()
    stats.probplot(data["SalePrice"], plot=plt)
    plt.show()
    sns.heatmap(corr, annot=True, cmap = plt.cm.seismic)
    plt.show()

# Train test split
x = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

predictions = lin_reg.predict(x_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"{mse = }")

cor_target = abs(corr["SalePrice"])
relevant_features = cor_target[cor_target > .2]
names = [index for index, _ in relevant_features.iteritems()]

names.remove("SalePrice")
print(data.shape[1]-1 - len(names), "removed feature")
