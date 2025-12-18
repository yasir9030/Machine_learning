import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('F:\\machine_learning\\Boston.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Correct train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Training set visualization
plt.figure(figsize=(8,6))
plt.scatter(X_train[:, 0], y_train, color='red')  # first feature
plt.plot(X_train[:, 0], regressor.predict(X_train), color='blue')
plt.title('Feature 0 vs Target (Training Set)')
plt.xlabel('Feature 0')
plt.ylabel('Target')
plt.show()


# Test set visualization
plt.figure(figsize=(8,6))
plt.scatter(X_test[:,0], y_test, color='red')
plt.plot(X_train[:,0], regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
"""***plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue')  # perfect prediction line
plt.title('Actual vs Predicted (Test Set)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()"""
