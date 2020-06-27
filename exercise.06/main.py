import math
import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt


def normal_distribution(x: float, mean: float, variance: float) -> float:
    pi_part = 1 / (variance * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mean) / variance) ** 2

    return pi_part * math.exp(exponent)


def generate_distribution(width: float, mean: float, variance: float) -> List[float]:
    points = []

    while len(points) < 100:
        x = random.uniform(-width, width)
        prop = normal_distribution(x, mean, variance)

        if random.random() < prop:
            points.append(x)

    return points


def generate_distribution_points(center: (float, float), variance: (float, float)) -> (List[float], List[float]):
    x_points = generate_distribution(8, center[0], variance[0])
    y_points = generate_distribution(8, center[1], variance[1])

    return x_points, y_points


random.seed(19980528)
x_points_1, y_points_1 = generate_distribution_points((5, -5), (1.3, 0.9))
x_points_2, y_points_2 = generate_distribution_points((-5, 2), (1.4, 1.4))
x_points_3, y_points_3 = generate_distribution_points((0, 6),  (1.0, 0.8))

plt.figure(0)
plt.scatter(x_points_1, y_points_1)
plt.scatter(x_points_2, y_points_2)
plt.scatter(x_points_3, y_points_3)
plt.savefig("ml.exercise.06.01.png", dpi=300)
