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
    # mean = np.array([0, 0])
    # sig = np.array([[0.5, 1.5], [0.1, 1.6]])
    mean = np.array([2.3338281687488074, -0.08093649607381477])
    sig = np.array([[29.313368698570045, 23.845319805965087], [23.845319805965087, 29.313368698570045]])

    sig_inv = np.linalg.inv(sig)
    sig_det = np.linalg.det(sig)

    exponent = -0.5 * np.transpose(x - mean) * sig * (x - mean)

    # return np.exp(exponent) / (math.sqrt((2 * math.pi) ** 2) * sig_det)
    return np.exp(exponent)


def f(x, y):
    data = np.zeros((len(x), len(x)))

    # data[:, :, 0] = x
    # data[:, :, 1] = y

    for inx in range(len(x)):
        for iny in range(len(x)):
            data[inx][iny] = np.mean(normal_distribution_2(np.array([x[inx][iny], y[inx][iny]])))

    # print("data", data)

    # print("normal_distribution(x, 1, 0, 1)", normal_distribution(x, 1, 0, 1))

    # return normal_distribution(x, 1, 0, 1) + normal_distribution(y, 1, 0, 1)
    # return data
    return data


Z = f(X, Y)

# Create a surface plot and projected filled contour plot under it.
plt.figure(0)
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,cmap=cm.viridis)

plt.contourf(X, Y, Z, 20)

# Adjust the limits, ticks and view angle
# ax.set_zlim(-0.15, 0.2)
# ax.set_zticks(np.linspace(0, 0.2, 5))
# ax.view_init(27, -21)

plt.show()

print("normal_distribution_2([0, 0])")
print(normal_distribution_2(np.array([0, 0])))
print("normal_distribution_2([1, 0])")
print(normal_distribution_2(np.array([1, 0])))
# print("normal_distribution_2([0, 0])", np.mean(normal_distribution_2(np.array([1, 0]))))
