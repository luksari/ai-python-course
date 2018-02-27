import numpy as np

iris = open("iris.data", 'r')

irisData = []
row = []

for line in iris:
    removedTrailing = line.strip("\r\n")
    row = removedTrailing.rsplit(",")
    irisData.append(row)

pet_len, pet_width, sep_len, sep_width = [], [], [], []
irisData = irisData[:-1]

for probe in irisData:
        pet_len.append(float(probe[2]))
        pet_width.append(float(probe[3]))
        sep_len.append(float(probe[0]))
        sep_width.append(float(probe[1]))


def showData(val, key):
    maxVal = max(val)
    minVal = min(val)
    avgVal = sum(val) / float(len(val))
    stdDevVal = np.std(val, axis=0)
    print("\n")
    print("Maximum " + key + " value: " + str(maxVal))
    print("Minimum " + key + " value: " + str(minVal))
    print("Average " + key + " value: " + str(avgVal)[:7])
    print("Standart deviation of " + key + " value: " + str(stdDevVal)[:7])


showData(pet_len, "petal length")
showData(pet_width, "petal width")
showData(sep_len, "sepal length")
showData(sep_width, "sepal width")
