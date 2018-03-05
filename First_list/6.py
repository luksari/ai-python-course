import numpy as np

iris = open("iris.data", 'r')

irisData = []
row = []

for line in iris:
    removedTrailing = line.strip("\r\n")
    row = removedTrailing.rsplit(",")
    irisData.append(row)

irisData = irisData[:-1]
_class, sep_len, sep_width, pet_len, pet_width = [], [], [], [], []

for probe in irisData:
    _class.append(probe[4])
    sep_len.append(float(probe[0]))
    sep_width.append(float(probe[1]))
    pet_len.append(float(probe[2]))
    pet_width.append(float(probe[3]))

data = np.column_stack((_class, sep_len, sep_width, pet_len, pet_width))

avg_pet_len = []
avg_pet_width = []
avg_sep_len = []
avg_sep_width = []


def showData(key):

    pet_len = []
    sep_len = []
    pet_width = []
    sep_width = []

    index = 0
    for val in data:
        if val[0] == key:
            sep_len.append(float(val[1]))
            sep_width.append(float(val[2]))
            pet_len.append(float(val[3]))
            pet_width.append(float(val[4]))

    avg_pet_len.append(sum(pet_len) / len(pet_len))
    avg_pet_width.append(sum(pet_width) / len(pet_width))
    avg_sep_width.append(sum(sep_width) / len(sep_width))
    avg_sep_len.append(sum(sep_len) / len(sep_len))

    if key == "Iris-setosa":
        index = 0
    elif key == "Iris-versicolor":
        index = 1
    elif key == "Iris-virginica":
        index = 2
    print("\n" + key + " average attributes values:"
          + "\npetal length: " + str(avg_pet_len[index])[:5]
          + "\npetal width: " + str(avg_pet_width[index])[:5]
          + "\nsepal length: " + str(avg_sep_len[index])[:5]
          + "\nsepal width: " + str(avg_sep_width[index])[:5])


showData("Iris-setosa")
showData("Iris-versicolor")
showData("Iris-virginica")
