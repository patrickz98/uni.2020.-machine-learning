# -*- coding: utf-8 -*-

import os
import math
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Set, Dict, Tuple, Optional

dataDir = "../exercise.03.data"


def img_to_feature_vector(img: np.ndarray) -> np.ndarray:
    red_min = np.min(img[:, :, 0])
    green_min = np.min(img[:, :, 1])
    blue_min = np.min(img[:, :, 2])

    red_mean = np.mean(img[:, :, 0])
    green_mean = np.mean(img[:, :, 1])
    blue_mean = np.mean(img[:, :, 2])

    # print("img", img[:, :, 0])

    return np.array([
        red_min,
        green_min,
        blue_min,
        red_mean,
        green_mean,
        blue_mean,
    ])


def mean_for_feature_vectors(vectors: [np.array]) -> np.array:
    feature_sum = np.zeros((6,))

    for vector in vectors:
        feature_sum += vector

    return feature_sum / len(vectors)


def calculate_feature_vectors(src: str) -> List[np.array]:
    vectors = []

    source = dataDir + "/" + src
    files = os.listdir(source)
    for file in files:
        if not file.endswith(".png"):
            continue

        path = source + "/" + file
        # print(path)

        img = skimage.io.imread(path)
        feature_vector = img_to_feature_vector(img)

        vectors += [feature_vector]

    return vectors


def probability(x: np.array, y: np.array, coefficient_matrix: np.array) -> float:

    n = len(x)

    print("n", n)
    print("x", x)
    print("reshape", x.reshape(-1, 1))

    # Initialize and reshape
    X = x.reshape(-1, 1)
    MU = y.reshape(-1, 1)
    p, _ = coefficient_matrix.shape

    SIGMA_inv = np.linalg.inv(coefficient_matrix)
    denominator = np.sqrt((2 * np.pi) ** (n / 2) * np.linalg.det(coefficient_matrix) ** (1 / 2))
    exponent = -(1 / 2) * ((X - MU).T @ SIGMA_inv @ (X - MU))

    return float((1. / denominator) * np.exp(exponent))


def main():
    negatives = calculate_feature_vectors("negatives")
    negative_mean = mean_for_feature_vectors(negatives)
    print("negative_mean", negative_mean)

    positives = calculate_feature_vectors("positives")
    positives_mean = mean_for_feature_vectors(positives)
    print("positives_mean", positives_mean)

    coefficient_matrix = np.zeros((6, 6))
    print(coefficient_matrix)

    for pos in positives:
        defuck = np.array([pos - positives_mean])
        coefficient_matrix += defuck * np.transpose(defuck)

    for neg in negatives:
        defuck = np.array([neg - negative_mean])
        coefficient_matrix += defuck * np.transpose(defuck)

    coefficient_matrix = coefficient_matrix / (len(negatives) + len(positives))
    print("coefficient_matrix", coefficient_matrix)
    print("det", np.linalg.det(coefficient_matrix))

    #img = skimage.io.imread(dataDir + "/positives/p02.png")
    img = skimage.io.imread(dataDir + "/negatives/n01.png")
    feature_vector = img_to_feature_vector(img)

    p_x_equals_y = probability(feature_vector, positives_mean, coefficient_matrix)
    p_x_not_equals_y = probability(feature_vector, negative_mean, coefficient_matrix)
    print("p_x_equals_y", "{:.20f}".format(p_x_equals_y))

    prob = (p_x_equals_y * 0.5) / (p_x_not_equals_y * 0.5 + p_x_equals_y * 0.5)
    print("probability", "{:.20f}".format(prob))


if __name__ == "__main__":
    main()
