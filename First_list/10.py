import numpy as np
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt

first_attr = np.random.randn(10)-2
second_attr = 10*np.random.randn(10)
data = np.vstack((first_attr, second_attr))
data=data.conj().transpose()

normalize = preprocessing.MinMaxScaler((0,1))
data = normalize.fit_transform(data)

euclidean_matrix = metrics.pairwise.pairwise_distances(
    data, metric='euclidean')
mahalanobian_matrix = metrics.pairwise.pairwise_distances(
    data,  metric='mahalanobis')
minkowskian_matrix = metrics.pairwise.pairwise_distances(
    data, metric='minkowski')

print("Macierz odległości euklidesowych:\n ")
print(euclidean_matrix)
print("Macierz odległości mahalanobisa:\n ")
print(mahalanobian_matrix)
print("Macierz odległości minkowskiego:\n ")
print(minkowskian_matrix)
