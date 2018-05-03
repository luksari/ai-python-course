from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSCanonical
from sklearn.neighbors import KNeighborsClassifier
import math
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

dataSet = datasets.load_digits()
data = dataSet["data"]
target = dataSet["target"]
 
plsca = PLSCanonical(n_components = 2)
plsca.fit(data,target)
 
X_train_r,Y_train_r = plsca.transform(data,target)
 
knn = math.sqrt(len(X_train_r))
knn = KNeighborsClassifier(n_neighbors = int(knn))
 
Y_train_r = [int(Y_train_r[i])for i in range(0,len(
Y_train_r))]
 
k = knn.fit(X_train_r,Y_train_r)
print(k.score(X_train_r,Y_train_r))
knn = KNeighborsClassifier(n_neighbors = 4)
 
sfs = SFS(knn,
    k_features = 3,
    forward = True,
    floating = False,
    verbose = 2,
    scoring = 'accuracy',
    cv = 0)

print(sfs)