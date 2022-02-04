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
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

x = df.values[:,:-1]
y = df.values[:,-1]

print (x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .7, random_state=0)

mlp_reg = MLPRegressor(random_state=0, max_iter=500).fit(x_train, y_train)
predictions = mlp_reg.predict(x_test)

print (mlp_reg.score(x_test, y_test))

### Build many models and compare them

from sklearn.model_selection import GridSearchCV # this will be used to train, test and compare multiple models
from sklearn.metrics import SCORERS, classification_report

params_to_try = {
    "hidden_layer_sizes": [(l1,l2) for l1 in [50, 100, 200, 500] for l2 in [1, 50, 100, 200, 500]],
    "learning_rate_init":[0.001,0.01,0.1],
    "max_iter": [500],
    "random_state": [0]
}

grid = GridSearchCV(MLPRegressor(), param_grid = params_to_try,n_jobs=-1, scoring="neg_mean_squared_error")

grid.fit(x_train, y_train)
print ("Best model found for paramaters:", grid.best_params_)

predictions = grid.predict(x_test)

from sklearn.metrics import mean_squared_error

print ("With best model, RMSE is",mean_squared_error(y_test, predictions))

#Build dataframe to better visualize results
results_dict = {
    "neurons": [i["hidden_layer_sizes"] for i in grid.cv_results_["params"]],
    "learning_rate": [i["learning_rate_init"] for i in grid.cv_results_["params"]],
    "mean_test_score" : grid.cv_results_["mean_test_score"],
}

print (pd.DataFrame(results_dict).sort_values("mean_test_score", ascending=False))