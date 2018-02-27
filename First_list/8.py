import numpy as np
import matplotlib.pyplot as plt

first_attr = np.random.rand(10) - 2
second_attr = np.random.uniform(0, 10, 10)
data = np.column_stack((first_attr, second_attr))

plt.plot(first_attr, second_attr, 'ro')
plt.xlabel("N(-2,1) Distribiution")
plt.ylabel("[0,10] Distribiution")
plt.show()
