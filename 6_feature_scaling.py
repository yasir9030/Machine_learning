import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv(r'F:\machine_learning\dataset_with_missing.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Original X:")
print(X)
print("\nOriginal y:")
print(y)

# Step 1: Imputation
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print("\nX after imputation:")
print(X)

# Step 2: OneHotEncoding
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

print("\nX after OneHotEncoding:")
print(X)

# Step 3: Train-Test Split
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

# ---------------------------
# ⭐ ADDING THIS PART (FEATURE SCALING)
# ---------------------------

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Scale all numerical columns (all except the first few OHE columns)
# The OHE columns are 0/1; scaling is optional but usually done
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])

print("\nX_train after Feature Scaling:")
print(X_train)

print("\nX_test after Feature Scaling:")
print(X_test)
