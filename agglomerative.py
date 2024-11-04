from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# Load your dataset
data = pd.read_csv('/content/drive/MyDrive/dataset1.csv')

X = data[['Milk', 'Grocery', 'Frozen']].values  # Update with relevant feature columns

# Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=4)
initial_clusters = agg_cluster.fit_predict(X)

# Plot dendrogram to visualize hierarchical clustering
Z = linkage(X, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram for Agglomerative Clustering')
plt.show()

# Final clusters
final_clusters = agg_cluster.labels_

# Error rate (for Agglomerative, this is typically not calculated like KMeans)
error_rate = 'Not applicable for Agglomerative'

# Output details
print("Final Clusters:", final_clusters)
print("Error Rate:", error_rate)

# Plot Final Clusters
plt.scatter(X[:, 0], X[:, 1], c=final_clusters, cmap='rainbow')
plt.title('Final Clusters after Agglomerative Clustering')
plt.show()
