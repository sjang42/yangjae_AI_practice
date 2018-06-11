# Task 1. K-means clustering from scratch

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from copy import deepcopy

## code for elice
# from elice_utils import EliceUtils
# eu = EliceUtils()

# Check versions
print('numpy version: ', np.__version__)
print('pandas version: ', pd.__version__)

np.random.seed(12345)

# Q1. Create a dataset (X and y) with 3 clusters using sklearn.datasets.make_blobs
X, y = make_blobs(n_samples=800, n_features=2, centers=3, random_state=12345)


# Q2: define a function to calculate Euclidean distance
def dist(a, b, axis=1):
    """
    :param a: 1-D input array
    :param b: 1-D input array
    :param axis: an integer for the axis of a and b along which to compute the vector norms
    :return: Eucleadian distance (float)
    """
    l2norm = np.linalg.norm(a - b, axis=axis)
    return l2norm


# Q3. Number of clusters
K = 3

# X coordinates of initial centroids
C_x = [-7, 0, 7]
# Y coordinates of initial centroids
C_y = [0, 0, 0]

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

print("initial centroids: ", C)

# Plotting along with the initial Centroids
plt.scatter(X[:, 0], X[:, 1], c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=150, c='g')

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)

# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))

# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
print('initial error: ', error)

# Loop will run till the error is less than the threshold value
while error >= 1e-4:
    # Q4. Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Q5. Finding the new centroids by taking the average value
    for i in range(K):
        points = X[np.where(clusters == i)[0]]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

print('final error: ', error)
print('final Centroid: ', C)

# Plotting along with the final Centroids
plt.scatter(C[:, 0], C[:, 1], marker='*', s=250, c='r')
plt.savefig("final_cluster.png")

# code for elice
# eu.send_image("final_cluster.png")
