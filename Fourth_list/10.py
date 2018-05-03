
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import math

with open('arcene_train.data') as f: raw_data = f.read()
 
data = np.loadtxt('arcene_train.data')
labels = np.loadtxt('arcene_train.labels')
 
train = data[int(0.7*len(data)):]
test = data[:int(0.3*len(data))]
labels = labels[int(0.7*len(data)):]
 
knn = KNeighborsClassifier(n_neighbors = 5)
 
sbs = SFS(knn,
    k_features = 20,
    forward = False,
    floating = False,
    scoring = 'accuracy',
    cv = 4)
   
sbs = sbs.fit(train,labels)
print(sbs.k_score_)