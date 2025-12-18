import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#
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


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

print("\nX_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)
