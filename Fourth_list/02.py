from sklearn import datasets
import numpy as np
import random
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
target_names = digits.target_names
X = digits.data
Y = digits.target
train, test, train_targets, test_targets = train_test_split(X, Y, test_size=0.5, random_state=42)
print(len(train), len(test))