import math
import random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


# https://www.youtube.com/watch?v=qMTuMa86NzU

# def normal_distribution(x: float, pi: float, mean: float, variance: float) -> float:
#     # pi_part = 1 / (variance * math.sqrt(2 * math.pi))
#     # pi_part = 1
#     pi_part = pi
#     exponent = -0.5 * ((x - mean) / variance) ** 2
#
#     return pi_part * math.exp(exponent)
#
#
# def generate_distribution(width: float, mean: float, variance: float) -> List[float]:
#     points = []
#
#     while len(points) < 100:
#         x = random.uniform(-width, width)
#         prop = normal_distribution(x, 1, mean, variance)
#
#         if random.random() < prop:
#             points.append(x)
#
#     return points

# def normal_distribution(x: np.array, mean: np.array, variance: np.array):
#     # mean = np.array([0, 0])
#     # sig = np.array([2, 1])
#
#     exponent = np.sum(((x - mean) ** 2) / variance)
#     result = math.exp(-exponent)
#     return result


def normal_distribution(x: np.array, mean: np.array, cov: np.array):
    cov_inv = np.linalg.inv(cov)
    exponent = np.transpose(x - mean) @ cov_inv @ (x - mean)
    result = math.exp(-exponent)

    return result


# def generate_distribution_points(center: np.array, variance: np.array) -> (List[float], List[float]):
#     x_points = normal_distribution(8, center, variance)
#     y_points = normal_distribution(8, center, variance)
#
#     return x_points, y_points


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


def em(k: int, training_data: List[Tuple[float, float, int]]):
    mean_c = []
    cov_c = []

    for _ in range(k):
        mean_x = random.uniform(-10, 10)
        mean_y = random.uniform(-10, 10)
        mean_c.append((mean_x, mean_y))

        cov = np.zeros((2, 2))
        cov[0, 0] = random.uniform(0.1, 3)
        cov[1, 1] = random.uniform(0.1, 3)
        cov_c.append(cov)

    for _ in range(2):
        probs_ic = []

        for i in range(len(training_data)):
            x = np.array(training_data[i])
            probs_ic.append([])

            for c in range(k):
                prob = normal_distribution(x, mean_c[c], cov_c[c])
                probs_ic[i].append(prob)

        r_ic = [[] for _ in range(len(training_data))]

        for i in range(len(probs_ic)):

            probs = probs_ic[i]

            sum = 0.0

            for prob in probs:
                print("prob", prob)
                sum += prob

            for c in range(k):
                probability = probs[c] / sum
                # print("probability", probability)
                r_ic[i].append(probability)

        mc = [0.0 for _ in range(k)]
        m = 0.0

        for c in range(k):

            for i in range(len(training_data)):
                mc[c] += r_ic[i][c]
                m += r_ic[i][c]

        mean_c = [[0.0, 0.0] for _ in range(k)]

        for c in range(k):
            for i in range(len(training_data)):
                x = np.array(training_data[i])
                print(f"r_ic[{i}][{c}] = {r_ic[i][c]}")
                print("x =", x)
                mean_c[c][0] += np.sum(r_ic[i][c] * x[0])
                mean_c[c][1] += np.sum(r_ic[i][c] * x[1])

            mean_c[c][0] *= (1 / mc[c])
            mean_c[c][1] *= (1 / mc[c])

        print(mean_c)

        cov_c = [np.zeros((2, 2)) for _ in range(k)]

        for c in range(k):
            for i in range(len(training_data)):
                np_x = np.array(training_data[i])
                cov_c[c] += r_ic[i][c] * (np.transpose(np_x) * np_x)

            cov_c[c] = (1 / mc[c]) * cov_c[c]
            cov_c[c][0, 1] = 0
            cov_c[c][1, 0] = 0
            print(f"cov_c[{c}] = {cov_c[c]}")

    return mean_c, cov_c


random.seed(19980528)

cov = [[1, -1],
       [0, 2]]
x_points_1, y_points_1 = np.random.multivariate_normal([8, 8], cov, 100).T
x_points_2, y_points_2 = np.random.multivariate_normal([-2, 0], cov, 100).T

# x_points_1, y_points_1 = generate_distribution_points((8, 8), (1.9, 1.9))
# x_points_2, y_points_2 = generate_distribution_points((-6, -3), (1.4, 2.4))
# x_points_3, y_points_3 = generate_distribution_points((4, -4), (2.8, 1.8))

plt.figure(0)
plt.scatter(x_points_1, y_points_1)
plt.scatter(x_points_2, y_points_2)
# plt.scatter(x_points_3, y_points_3)
# plt.savefig("ml.exercise.06.01.png", dpi=300)

training_data = []

for inx, iny in zip(x_points_1, y_points_1):
    training_data.append((inx, iny))

for inx, iny in zip(x_points_2, y_points_2):
    training_data.append((inx, iny))

# for inx, iny in zip(x_points_3, y_points_3):
#     training_data.append((inx, iny, 2))

# centroids = k_mean(3, training_data)
centroids, covs = em(2, training_data)

for inx in range(len(centroids)):
    centroid = centroids[inx]
    cov = covs[inx]
    print("centroid", centroid)
    print("cov", cov)
    plt.scatter(centroid[0], centroid[1], marker="x")

plt.savefig("ml.exercise.06.01.png", dpi=300)

# print("centroids", centroids)
# print("sig", sig)
