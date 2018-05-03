from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
import math
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

data = np.loadtxt('arcene_train.data')
labels = np.loadtxt('arcene_train.labels')
 
random.shuffle(data)
 
train = data[int(0.7*len(data)):]
test = data[:int(0.3*len(data))]
 
train = np.array(data[int(0.7*len(data)):])
train_labels = np.array(labels[int(0.7*len(data)):])
test = np.array(data[:int(0.3*len(data))])
 
knn = KNeighborsClassifier(n_neighbors = 4)
 
sfs = SFS(knn,
    k_features = math.sqrt(len(train)),
    forward = True,
    floating = False,
    scoring = 'accuracy',
    cv = 4)
 

sfs.fit(train, train_labels)
print("SFS: ", sfs.k_score_)