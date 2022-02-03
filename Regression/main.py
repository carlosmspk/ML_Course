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
X, y = load_boston.data, load_boston.target

data = pd.DataFrame(X, columns=load_boston.feature_names)
data["SalePrice"] = y
print(data.shape)
print(data.info())

if show_plots:
    # We can inspect every single variable's relationship to all others. But in large amounts of features, that's hardly practical
    # sns.pairplot(data, height=2.5)
    # plt.tight_layout()
    # plt.show()
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
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

predictions = lin_reg.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"{mse = }")

cor_target = abs(corr["SalePrice"])
relevant_features = cor_target[cor_target > .2]
names = [index for index, _ in relevant_features.iteritems()]

names.remove("SalePrice")
print(data.shape[1]-1 - len(names), "removed feature")

# we can then use the "new" dataset with the removed feature and we get one less "noisy" variable that is introducing meaningless data
