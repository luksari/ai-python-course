iris = open("iris.data", 'r')

irisData = []

for line in iris:
        row = line.split(",")
        irisData.append(row)

print("Number of probes: " + " " + str(len(irisData)))

for probe in irisData:
        print("Number of attributes: " + str(len(probe)))
