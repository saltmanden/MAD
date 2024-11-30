import numpy as np
from linweighre import LinearRegression
raw = np.genfromtxt('men-olympics-100.txt',delimiter=' ')
X = raw[:,0]
y = raw[:,1]

y = y.reshape((len(y), 1))

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.plot(y, model, color = "red")
plt.show()