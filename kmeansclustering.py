import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from google.colab import drive
drive.mount('/content/drive')

# Load your dataset
data = pd.read_csv('/content/drive/MyDrive/dataset1.csv')

# Inspect column names and replace these with actual feature names
X = data[['Milk', 'Grocery', 'Frozen']].values  # Update with relevant feature columns

# K-Means clustering
kmeans = KMeans(n_clusters=4, init='random', max_iter=300, random_state=42)
initial_clusters = kmeans.fit_predict(X)

# Get the final clusters
final_clusters = kmeans.labels_
epoch_size = kmeans.n_iter_
error_rate = kmeans.inertia_

# Plot Initial Clusters (assuming 2D for visualization, modify if needed)
plt.scatter(X[:, 0], X[:, 1], c=initial_clusters, cmap='rainbow')
plt.title('Initial Clusters')
plt.show()

# Output details
print("Final Clusters:", final_clusters)
print("Epoch Size (Iterations):", epoch_size)
print("Final Error Rate (Inertia):", error_rate)

# Plot Final Clusters
plt.scatter(X[:, 0], X[:, 1], c=final_clusters, cmap='rainbow')
plt.title('Final Clusters after K-Means')
plt.show()
