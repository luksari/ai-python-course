import matplotlib.pyplot as plt

iris = open("iris.data", 'r')

irisData = []
row = []

for line in iris:
    removedTrailing = line.strip("\r\n")
    row = removedTrailing.rsplit(",")
    irisData.append(row)

irisData = irisData[:-1]
pet_len, sep_len = [], []
p_lenSet, p_lenVer, p_lenVir, s_lenSet, s_lenVer, s_lenVir = [], [], [], [], [], []

for probe in irisData:
    pet_len.append(float(probe[0]))
    sep_len.append(float(probe[2]))
    if probe[4] == "Iris-setosa":
        p_lenSet.append(float(probe[2]))
        s_lenSet.append(float(probe[0]))
    elif probe[4] == "Iris-versicolor":
        p_lenVer.append(float(probe[2]))
        s_lenVer.append(float(probe[0]))
    elif probe[4] == "Iris-virginica":
        s_lenVir.append(float(probe[0]))
        p_lenVir.append(float(probe[2]))
max_sep_len = max(sep_len)
max_pet_len = max(pet_len)
min_pet_len = min(pet_len)
min_sep_len = min(sep_len)
offset = 0.25

plt.plot(p_lenSet, s_lenSet, 'ro')
plt.plot(p_lenVer, s_lenVer, 'go')
plt.plot(p_lenVir, s_lenVir, 'bo')
plt.ylabel("Petal length")
plt.xlabel("Sepal length")
plt.show()
