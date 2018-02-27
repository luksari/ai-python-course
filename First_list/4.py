import matplotlib.pyplot as plt

iris = open("iris.data", 'r')

irisData = []
row = []

for line in iris:
    removedTrailing = line.strip("\r\n")
    row = removedTrailing.rsplit(",")
    irisData.append(row)

sep_len = []
sep_width = []
irisData = irisData[:-1]

for probe in irisData:
        sep_len.append(float(probe[0]))
        sep_width.append(float(probe[1]))

plt.plot(sep_len, sep_width, 'ro')
plt.ylabel("Sepal width")
plt.xlabel("Sepal length")
plt.show()
