from sklearn.model_selection import train_test_split
import numpy as np
import random

with open('arcene_train.data') as f: raw_data = f.read()
 
data = np.loadtxt('./arcene_train.data')
 
random.shuffle(data)
 
train = data[int(0.7*len(data)):]
test = data[:int(0.3*len(data))]

print(len(train), len(test))