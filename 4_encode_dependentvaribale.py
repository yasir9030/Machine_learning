import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv(r'F:\machine_learning\dataset_with_missing.csv')


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Original X:")
print(X)
print("\nOriginal y:")
print(y)


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])         
X[:, 1:3] = imputer.transform(X[:, 1:3])

print("\nX after imputation:")
print(X)


ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

print("\nX after OneHotEncoding:")
print(X)
le= LabelEncoder()
y=le.fit_transform(y)
print (y)