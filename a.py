import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv("/Users/kuldeeppareta/Desktop/churn_modeling/data")

print(df.head())





result = df.groupby("Geography")["CustomerId"].count()

print(result)

result = df.groupby("Gender")["CustomerId"].count()

print(result)


from sklearn.preprocessing import LabelEncoder



df.drop("Surname", axis=1, inplace=True)


le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Geography"] = le.fit_transform(df["Geography"])


print(df.head())

print(df.dtypes)


mm = MinMaxScaler()

X = df.drop(columns=['Exited'])
y = df['Exited']

from sklearn.model_selection import train_test_split

mm = MinMaxScaler()


X_mm = mm.fit_transform(X)

X_new = pd.DataFrame(
    X_mm,
    columns=X.columns,
    index=X.index
)


print(np.round(X_new.describe(), 2))


