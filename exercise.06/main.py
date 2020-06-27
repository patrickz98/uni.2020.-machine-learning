import math
import random
import numpy as np
import matplotlib.pyplot as plt


def normal_distribution(x: float, mean: float, variance: float) -> float:
    pi_part = 1 / (variance * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mean) / variance) ** 2

    return pi_part * math.exp(exponent)


x_points = [random.uniform(-5, 5) for x in range(20)]
x_points.sort()

y_points = [normal_distribution(x, 0, 1) for x in x_points]

plt.figure(0)
# plt.plot(x_points, y_points)
plt.scatter(x_points, y_points)
plt.savefig("ml.exercise.06.01.png", dpi=400)
