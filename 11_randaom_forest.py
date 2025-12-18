import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv('F:\\machine_learning\\random_forest_regression_data.csv')

x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

# Train Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)



# =====================================================
# PLOT 1 : Feature 1 vs Target  (Feature 2 fixed at MIN)
# =====================================================

fixed_feature2 = min(x[:, 1])   # use max(x[:,1]) for MAX

x1_grid = np.arange(min(x[:, 0]), max(x[:, 0]), 0.1)
x1_input = np.array([[val, fixed_feature2] for val in x1_grid])

plt.scatter(x[:, 0], y, color='red')
plt.plot(x1_grid, regressor.predict(x1_input), color='blue')
plt.title('Random Forest Regression (Feature 1)')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.show()

# =====================================================
# PLOT 2 : Feature 2 vs Target (Feature 1 fixed at MIN)
# =====================================================

fixed_feature1 = min(x[:, 0])   # use max(x[:,0]) for MAX

x2_grid = np.arange(min(x[:, 1]), max(x[:, 1]), 0.1)
x2_input = np.array([[fixed_feature1, val] for val in x2_grid])

plt.scatter(x[:, 1], y, color='green')
plt.plot(x2_grid, regressor.predict(x2_input), color='blue')
plt.title('Random Forest Regression (Feature 2)')
plt.xlabel('Feature 2')
plt.ylabel('Target')
plt.show()
