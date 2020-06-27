import math
import random
import numpy as np
import matplotlib.pyplot as plt


def normal_distribution(x: float, mean: float, variance: float) -> float:
    pi_part = 1 / (variance * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mean) / variance) ** 2

    return pi_part * math.exp(exponent)


x_points = []
y_points = []

for x in np.arange(-10, 10, 0.01):
    prop = normal_distribution(x, 0, 1)

    print("x", x)
    print("prop", prop)
    # print("random", random.uniform(0, 0.4))

    if random.random() < prop:
        x_points.append(x)
        y_points.append(prop)

plt.figure(0)
# plt.plot(x_points, y_points)
plt.scatter(x_points, y_points)
plt.savefig("ml.exercise.06.01.png", dpi=400)
