import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-10, 10, N)
Y = np.linspace(-10, 10, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 1.])
Sigma = np.array([[1., 1.], [1., 1.]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y


def normal_distribution(x: np.array, pi: float, mean: float, variance: float) -> float:
    # pi_part = 1 / (variance * math.sqrt(2 * math.pi))
    # pi_part = 1
    pi_part = pi
    exponent = -0.5 * ((x - mean) / variance) ** 2

    return pi_part * np.exp(exponent)


def normal_distribution_2(x: np.array):
    mean = np.array([-1.9627000753720651, -0.6402175481784353])
    # sig = np.array([2, 1])
    # cov = np.array([[1.0, -1.0],
    #                 [0.0, 2.0]])
    cov = np.array([[4.56755595, 0.0],
                    [0.0, 1.77421897]])

    cov_inv = np.linalg.inv(cov)
    # cov_det = np.linalg.det(cov)

    exponent = np.transpose(x - mean) @ cov_inv @ (x - mean)
    # exponent = (((x[0] - mean[0]) ** 2) / sig[0]) + (((x[1] - mean[1]) ** 2) / sig[1])

    # xx = x[0] - mean[0]
    # yy = x[1] - mean[1]
    #
    # exponent = (cov[0, 0] * (xx ** 2)) + (2 * cov[0, 1] * xx ** yy) + (cov[1, 1] * (yy ** 2))
    # exponent = np.sum(((x - mean) ** 2) / sig)

    print("x", x)
    print("exponent", exponent)

    result = math.exp(-exponent)
    print("result", result)

    return result


def f(x, y):
    data = np.zeros((len(x), len(x)))

    # data[:, :, 0] = x
    # data[:, :, 1] = y

    for inx in range(len(x)):
        for iny in range(len(x)):
            data[inx][iny] = normal_distribution_2(np.array([x[inx][iny], y[inx][iny]]))

    # print("data", data)

    # print("normal_distribution(x, 1, 0, 1)", normal_distribution(x, 1, 0, 1))

    # return normal_distribution(x, 1, 0, 1) + normal_distribution(y, 1, 0, 1)
    # return data
    return data


Z = f(X, Y)

# Create a surface plot and projected filled contour plot under it.
plt.figure(0)
plt.contourf(X, Y, Z, 100)
plt.show()

print("normal_distribution_2([0, 0])")
print(normal_distribution_2(np.array([0, 0])))
print("normal_distribution_2([1, 0])")
print(normal_distribution_2(np.array([1, 0])))
# print("normal_distribution_2([0, 0])", np.mean(normal_distribution_2(np.array([1, 0]))))
