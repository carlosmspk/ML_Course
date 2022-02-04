import pandas as pd

# store dataset as Data Frame (table format)
df = pd.read_csv("DeepLearning/dataset/WineQT.csv")
print ("First 5 rows of data:\n",df.head())