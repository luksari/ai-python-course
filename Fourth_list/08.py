from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
import math

data = np.loadtxt('arcene_train.data')
labels = np.loadtxt('arcene_train.labels')
 
random.shuffle(data)
 
train = data[int(0.7*len(data)):]
test = data[:int(0.3*len(data))]
 
train = np.array(data[int(0.7*len(data)):])
train_labels = np.array(labels[int(0.7*len(data)):])
test = np.array(data[:int(0.3*len(data))])
 
knn = KNeighborsClassifier(n_neighbors = 4)

sffs = SFS(knn, k_features=(1, 100), forward=True,
 floating=True,scoring="accuracy", cv=0) 