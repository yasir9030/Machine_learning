"""Linear Regression is a supervised machine learning algorithm used to predict a continuous numeric value.
It finds the best-fitting straight line that shows the relationship between input (X) and output (y).
For one input feature:
y=b0+b1x
Where:
y = predicted output
x = input value
b₀ = intercept (where line crosses y-axis)
b₁ = slope (how much y changes when x increases by 1)

"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('F:\\machine_learning\\data.csv')
x= dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#splitingdata set into traing set and test
x_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
#training the simple linear regression model on the training set
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('F:\\machine_learning\\salary.csv')
x= dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#splitingdata set into traing set and test
x_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
#training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
#predicitng the training set result 
y_pred=regressor.predict(X_test)
#visualising the training set result 
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience(training set)')
plt.xlabel("year of experience ")
plt.ylabel("salary")
plt.show()
#visualising the test set result 
plt.scatter(X_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel("year of experience ")
plt.ylabel("salary")
plt.show()