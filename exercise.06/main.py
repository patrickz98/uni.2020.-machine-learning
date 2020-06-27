import math
import random
from typing import List, Tuple
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


def k_mean(k: int, training_data: List[Tuple[float, float, int]]) -> List[Tuple[float, float]]:
    centroids = []
    c_values = []

    for _ in range(k):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        centroids.append((x, y))
        c_values.append([])

    for _ in range(1000):
        for inx in range(len(training_data)):
            x = training_data[inx]
            best_distance = -1
            best_centroid = -1

            for iny in range(k):
                centroid = centroids[iny]
                distance = (x[0] - centroid[0]) ** 2 + (x[1] - centroid[1]) ** 2

                # print(iny, "distance", distance)

                if best_distance < 0 or best_distance > distance:
                    best_distance = distance
                    best_centroid = iny

            c_values[best_centroid].append(x)

        for inx in range(k):
            x_sum = 0
            y_sum = 0

            count = len(c_values[inx])

            if count <= 0:
                continue

            for x in c_values[inx]:
                x_sum += x[0]
                y_sum += x[1]

            centroids[inx] = (x_sum / count, y_sum / count)
            c_values[inx] = []

    return centroids


random.seed(19980528)
x_points_1, y_points_1 = generate_distribution_points((8, 8), (1.9, 1.9))
x_points_2, y_points_2 = generate_distribution_points((-6, -3), (1.4, 2.4))
x_points_3, y_points_3 = generate_distribution_points((4, -4), (2.8, 1.8))

plt.figure(0)
plt.scatter(x_points_1, y_points_1)
plt.scatter(x_points_2, y_points_2)
plt.scatter(x_points_3, y_points_3)
# plt.savefig("ml.exercise.06.01.png", dpi=300)


training_data = []

for inx, iny in zip(x_points_1, y_points_1):
    training_data.append((inx, iny, 0))

for inx, iny in zip(x_points_2, y_points_2):
    training_data.append((inx, iny, 1))

for inx, iny in zip(x_points_3, y_points_3):
    training_data.append((inx, iny, 2))

centroids = k_mean(3, training_data)

for centroid in centroids:
    plt.scatter(centroid[0], centroid[1])
plt.savefig("ml.exercise.06.01.png", dpi=300)
