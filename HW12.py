import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate data
Nsamples = 500
C = 4
clusterSTD = 0.5
X,Y = make_blobs(n_samples=Nsamples, centers=C, cluster_std=clusterSTD, random_state=0)

# Use KMeans 
N = 4
K = KMeans(n_clusters=N, n_init=10, random_state=0)

labels = K.fit_predict(X)

# Plot the clusters
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, N))
for i in range(N):
    ax.scatter(X[labels == i, 0], X[labels == i, 1], 
            s=50, alpha=0.7, label=f'Cluster {i+1}', c=[colors[i]])

ax.scatter(K.cluster_centers_[:, 0], K.cluster_centers_[:, 1], 
           s=100, marker='o', color='black', label='Centroids')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'KMeans Clustering with {N} Clusters')
ax.legend()
plt.show()
