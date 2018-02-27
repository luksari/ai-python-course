import math as m

iris = open("iris.data", 'r')

irisData = []

for line in iris:
        row = line.split(",")
        irisData.append(row)

sum = 0

for i in range(3):
        dif = float(irisData[75][i]) - float(irisData[10][i])
        powDif = m.pow(dif, 2)
        sum = sum + powDif

sqrtSum = m.sqrt(sum)

print("Odleglosc euklidesowa: " + str(sqrtSum))
