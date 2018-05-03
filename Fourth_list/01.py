from sklearn import datasets
import numpy as np
from sklearn.decomposition import PCA

olivetti = datasets.fetch_olivetti_faces()
X = olivetti.data
Y = olivetti.target
max = 0
max_index = 0

for i in range(1, 7):
    pca = PCA(n_components=i)
    X_r = pca.fit(X).transform(X)
    print(i, ":", pca.explained_variance_ratio_.sum())
    if max < pca.explained_variance_ratio_.sum():
        max = pca.explained_variance_ratio_.sum()
        max_index = i
print("Best: ", max_index)