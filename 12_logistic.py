
# Logistic Regression with One-Hot Encoding for Gender (Index 1)

# 1️⃣ Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# 2️⃣ Load dataset
dataset = pd.read_csv('F:\\machine_learning\\Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values  # Features: Gender, Age, EstimatedSalary
y = dataset.iloc[:, -1].values   # Target: Purchased

# 3️⃣ One-Hot Encode Gender (index 1)
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), [1])],  # Encode Gender only
    remainder='passthrough'  # Keep Age and Salary as-is
)
X = ct.fit_transform(X)

# Convert all to float for StandardScaler
X = X.astype(float)

# 4️⃣ Split dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 5️⃣ Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6️⃣ Train Logistic Regression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 7️⃣ Make predictions and evaluate
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# 8️⃣ Visualization function
# -----------------------------
def plot_decision_boundary(X_set, y_set, title):
    # Plotting Age vs EstimatedSalary only (last two columns)
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, -2].min() - 1, stop=X_set[:, -2].max() + 1, step=0.01),
        np.arange(start=X_set[:, -1].min() - 1, stop=X_set[:, -1].max() + 1, step=0.01)
    )
    
    # Create grid array for prediction
    grid = np.zeros((X1.ravel().shape[0], X_set.shape[1]))
    grid[:, -2:] = np.array([X1.ravel(), X2.ravel()]).T  # Age and Salary
    # Gender column left as 0 (first category)

    plt.contourf(
        X1, X2,
        classifier.predict(grid).reshape(X1.shape),
        alpha=0.75,
        cmap=ListedColormap(("red", "green"))
    )
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    # Plot actual points
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, -2],  # Age
            X_set[y_set == j, -1],  # Estimated Salary
            c=ListedColormap(("red", "green"))(i),
            label=j
        )
    
    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.legend()
    plt.show()

# 9️⃣ Visualize Training set
plot_decision_boundary(X_train, y_train, "Logistic Regression (Training set)")

# 10️⃣ Visualize Test set
plot_decision_boundary(X_test, y_test, "Logistic Regression (Test set)")
