import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv('F:\\machine_learning\\svr_dataset.csv.')

x = dataset.iloc[:,1 :-1].values 
y = dataset.iloc[:, -1].values
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x, y)
print(regressor.predict([[6.5]]))

# Visualization
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (random forest regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

