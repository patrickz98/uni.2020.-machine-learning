import math
import random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


# https://www.youtube.com/watch?v=qMTuMa86NzU

def normal_distribution(x: float, pi: float, mean: float, variance: float) -> float:
    # pi_part = 1 / (variance * math.sqrt(2 * math.pi))
    # pi_part = 1
    pi_part = pi
    exponent = -0.5 * ((x - mean) / variance) ** 2

    return pi_part * math.exp(exponent)


def generate_distribution(width: float, mean: float, variance: float) -> List[float]:
    points = []

    while len(points) < 100:
        x = random.uniform(-width, width)
        prop = normal_distribution(x, 1, mean, variance)

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


def em(k: int, training_data: List[Tuple[float, float, int]]):
    mean_c = []
    sig_c = []
    pi_c = []

    for _ in range(k):
        mean_x = random.uniform(-10, 10)
        mean_y = random.uniform(-10, 10)
        mean_c.append((mean_x, mean_y))

        sig_x = random.uniform(0.1, 5)
        sig_y = random.uniform(0.1, 5)
        sig_c.append((sig_x, sig_y))

        pi_c.append((1, 1))

    probs_ic = []

    for i in range(len(training_data)):
        x = training_data[i]
        probs_ic.append([])

        for c in range(k):
            prob_x = normal_distribution(x[0], pi_c[c][0], mean_c[c][0], sig_c[c][0])
            prob_y = normal_distribution(x[1], pi_c[c][1], mean_c[c][1], sig_c[c][1])
            probs_ic[i].append((prob_x, prob_y))

    r_ic = [[] for _ in range(len(training_data))]

    for i in range(len(probs_ic)):

        probs = probs_ic[i]

        sum_x = 0.0
        sum_y = 0.0

        for prob in probs:
            print("prob", prob)

            sum_x += prob[0]
            sum_y += prob[1]

        for c in range(k):
            x_prob = probs[c][0] / sum_x
            y_prob = probs[c][1] / sum_y
            # r_ic[i].append((x_prob, y_prob))

            probability = (x_prob + y_prob) / 2
            # print("probability", probability)
            r_ic[i].append(probability)

    mc = [[0.0, 0.0] for _ in range(k)]
    m = [0.0, 0.0]

    for c in range(k):

        for i in range(len(training_data)):
            mc[c][0] += r_ic[i][c]
            mc[c][1] += r_ic[i][c]
            m[0] += r_ic[i][c]
            m[1] += r_ic[i][c]

    max_pi_c = []

    for c in range(k):
        max_pi_c.append((mc[c][0] / m[0], mc[c][1] / m[1]))

    print("max_pi_c", max_pi_c)

    max_mean_c = [[0.0, 0.0] for _ in range(k)]

    for c in range(k):
        for i in range(len(training_data)):
            x = training_data[i]
            max_mean_c[c][0] += r_ic[i][c] * x[0]
            max_mean_c[c][1] += r_ic[i][c] * x[1]

        max_mean_c[c][0] *= (1 / mc[c][0])
        max_mean_c[c][1] *= (1 / mc[c][1])

    print(max_mean_c)

    max_sig = [[0.0, 0.0] for _ in range(k)]

    for c in range(k):
        for i in range(len(training_data)):
            x = training_data[i]
            max_sig[c][0] += r_ic[i][c] * (x[0] - max_mean_c[c][0]) ** 2
            max_sig[c][1] += r_ic[i][c] * (x[1] - max_mean_c[c][1]) ** 2

        max_sig[c][0] *= 1 / mc[c][0]
        max_sig[c][1] *= 1 / mc[c][1]

    return max_mean_c, max_sig


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

# centroids = k_mean(3, training_data)
centroids, sig = em(3, training_data)

for centroid in centroids:
    plt.scatter(centroid[0], centroid[1], marker="x")

plt.savefig("ml.exercise.06.01.png", dpi=300)

print("centroids", centroids)
print("sig", sig)
