import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('F:\\machine_learning\\Mall_Customers.csv')

# Select features (Annual Income & Spending Score)
x = dataset.iloc[:, [3, 4]].values

# Dendrogram
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(8, 5))
sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Agglomerative Hierarchical Clustering (3 clusters)
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(
    n_clusters=3,
    metric='euclidean',
    linkage='ward'
)

y_hc = hc.fit_predict(x)

# Visualising the 3 clusters
plt.figure(figsize=(8, 5))

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1],
            s=100, c='red', label='Cluster 1')

plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1],
            s=100, c='blue', label='Cluster 2')

plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1],
            s=100, c='green', label='Cluster 3')

plt.title('Customer Segmentation (3 Clusters)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.show()
