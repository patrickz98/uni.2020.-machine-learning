import math
import random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


# https://www.youtube.com/watch?v=qMTuMa86NzU


def f(x, y, mean: np.array, cov: np.array):
    data = np.zeros((len(x), len(x)))

    for inx in range(len(x)):
        for iny in range(len(x)):
            x_val = np.array([x[inx][iny], y[inx][iny]])
            data[inx][iny] = normal_distribution(x_val, mean, cov)

    return data


def show_gaussian(plt, means: np.array, covs: np.array):
    N = 60
    X = np.linspace(-10, 10, N)
    Y = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(X, Y)

    heatmap = np.zeros((len(X), len(Y)))
    for c in range(len(means)):
        mean = means[c]
        cov = covs[c]

        heatmap += f(X, Y, mean, cov)

        plt.plot(mean[0], mean[1], "x")

    plt.contour(X, Y, heatmap)


def normal_distribution(x: np.array, mean: np.array, cov: np.array):
    # cov_inv = np.linalg.inv(cov)

    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    exponent = np.transpose(x - mean) @ cov_inv @ (x - mean)
    result = math.exp(-exponent) / math.sqrt(((2 * math.pi) ** 2) * cov_det)

    return result


def k_mean(k: int, training_data: np.array) -> List[Tuple[float, float]]:
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


def em(k: int, training_data: np.array):
    mean_c = np.zeros((k, 2))
    cov_c = []

    for c in range(k):
        mean_x = random.uniform(-6, 6)
        mean_y = random.uniform(-6, 6)
        mean_c[c] = np.array([mean_x, mean_y])

        cov = np.zeros((2, 2))
        cov[0, 0] = random.uniform(0.5, 3)
        cov[1, 1] = random.uniform(0.5, 3)
        cov_c.append(cov)

    m = training_data.shape[0]

    for _ in range(100):
        r_ic = np.zeros((m, k))

        for i in range(m):
            x = training_data[i]
            for c in range(k):
                probability = normal_distribution(x, mean_c[c], cov_c[c])
                r_ic[i, c] = probability

            r_ic[i] = r_ic[i] / np.sum(r_ic[i])

        mc = np.zeros((k, 1))
        for c in range(k):
            mc[c] = np.sum(r_ic[:, c])

        mean_c = np.zeros((k, 2))
        for c in range(k):

            parts = []

            for i in range(m):
                x = training_data[i]
                parts.append(r_ic[i, c] * x)

            mean_c[c] = np.sum(parts, axis=0) / mc[c]

        for c in range(k):
            parts = []

            for i in range(m):
                x = training_data[i]
                xx = r_ic[i, c] * np.transpose([x - mean_c[c]]) * ([x - mean_c[c]])
                parts.append(xx)

            cov_c[c] = (1 / mc[c]) * np.sum(parts, axis=0)

    return mean_c, cov_c


random.seed(19980528)

mean1 = (0, 0)
cov1 = [[1.0, 0.5],
        [0.5, 1.0]]
data1 = np.random.multivariate_normal(mean1, cov1, 100)

mean2 = (3, 4)
cov2 = [[1.0, -0.7],
        [-0.7, 1.0]]
data2 = np.random.multivariate_normal(mean2, cov2, 100)

mean3 = (-2, 4)
cov3 = [[1.0, 0.9],
        [0.9, 1.0]]
data3 = np.random.multivariate_normal(mean3, cov3, 100)

# print("data1", data1)

training_data = np.concatenate((data1, data2, data3))

plt.figure(0)
plt.plot(data1[:, 0], data1[:, 1], "x")
plt.plot(data2[:, 0], data2[:, 1], 'x')
plt.plot(data3[:, 0], data3[:, 1], 'x')

# centroids = k_mean(3, training_data)
centroids, covs = em(3, training_data)

for inx in range(len(centroids)):
    centroid = centroids[inx]
    print(f"centroid[{inx}]")
    print(centroid)

    cov = covs[inx]
    print(f"covs[{inx}]")
    print(cov)

    plt.plot(centroid[0], centroid[1], "o")

# plt.savefig("ml.exercise.06.k_mean.png", dpi=300)
plt.savefig("ml.exercise.06.em.points.png", dpi=300)

show_gaussian(plt, centroids, covs)
plt.savefig(f"ml.exercise.06.em.gaussian.png", dpi=300)
