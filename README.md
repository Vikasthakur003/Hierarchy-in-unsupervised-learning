# Hierarchy-in-unsupervised-learning
Hierarchical Clustering using Dendrogram (Python)
## Overview

This project demonstrates Hierarchical Clustering using Python.
The goal is to group similar data points and visualize their relationships using a dendrogram.

Hierarchical clustering is an unsupervised machine learning algorithm, meaning it works without labeled data and helps discover natural groupings in the dataset.

## What This Code Does

Creates a small 2D dataset

Converts it into a Pandas DataFrame

Applies Agglomerative Hierarchical Clustering

Visualizes the hierarchy using a dendrogram

## Technologies Used

Python

NumPy – for numerical data handling

Pandas – for data manipulation

Matplotlib – for visualization

SciPy – for hierarchical clustering algorithms

## Dataset Description

The dataset contains 2D points:

(1, 2)
(2, 3)
(3, 4)
(8, 8)
(9, 9)
(10, 10)


These points naturally form two clusters:

Cluster 1: points close to each other near (1–4)

Cluster 2: points close to each other near (8–10)

## How the Algorithm Works

Each data point starts as its own cluster

The algorithm repeatedly:

Finds the closest clusters

Merges them together

This continues until all points form one hierarchy

The Ward method is used to minimize variance between clusters

## Visualization: Dendrogram

A dendrogram is a tree-like diagram that shows:

How clusters are merged

The distance between clusters

The hierarchical structure of the data

By cutting the dendrogram at a certain height, you can decide the number of clusters.

## Code Explanation
##  Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

## Create Dataset
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [8, 8],
    [9, 9],
    [10, 10]
])

## Convert to DataFrame
df = pd.DataFrame(X)
print(df)

## Apply Hierarchical Clustering
linked = linkage(df, method="ward")

## Plot the Dendrogram
plt.figure(figsize=(5,4))
dendrogram(linked)
plt.show()
## Output

A dendrogram showing two major clusters

Clear visual separation between similar data points

### When to Use Hierarchical Clustering

When you don’t know the number of clusters

When you want a visual explanation

For small to medium-sized datasets

For customer segmentation, document clustering, gene analysis, etc.

## Future Improvements

Automatically extract clusters using fcluster

Apply the algorithm to a real-world dataset

Compare with DBSCAN or K-Means
