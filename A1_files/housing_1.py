import numpy

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
mean_compute = t_train.mean()

# (b) RMSE function
def rmse(y_true, y_pred):
    return numpy.sqrt(numpy.mean((y_true - y_pred) ** 2))

rmse_loss = rmse(t_test, mean_compute)

# (c) visualization of results

import matplotlib.pyplot as plt

plt.scatter(range(len(t_test)), t_test)
plt.plot(range(len(t_test)), [mean_compute] * len(t_test), color = "red")
plt.show()