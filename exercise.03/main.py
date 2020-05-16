# -*- coding: utf-8 -*-

import os
import math
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Set, Dict, Tuple, Optional

dataDir = "../exercise.03.data"


# For each one of the 60 images obtain a feature vector.
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


def probability(x: np.array, mean: np.array, coefficient_matrix: np.array) -> float:

    n = len(x)

    x = x.reshape(-1, 1)
    mean = mean.reshape(-1, 1)
    p, _ = coefficient_matrix.shape

    sigma_inv = np.linalg.inv(coefficient_matrix)
    denominator = np.sqrt((2 * np.pi) ** (n / 2) * np.linalg.det(coefficient_matrix) ** (1 / 2))
    exponent = -(1 / 2) * ((x - mean).T @ sigma_inv @ (x - mean))

    return float((1. / denominator) * np.exp(exponent))


def classify_image(path: str, pos_mean: np.array, neg_mean: np.array, coefficient_matrix: np.array):

    img = skimage.io.imread(path)
    feature_vector = img_to_feature_vector(img)

    p_x_equals_y = probability(feature_vector, pos_mean, coefficient_matrix)
    p_x_not_equals_y = probability(feature_vector, neg_mean, coefficient_matrix)

    prob = (p_x_equals_y * 0.5) / (p_x_not_equals_y * 0.5 + p_x_equals_y * 0.5)

    print("Probability for having a Chagas parasites:    ", "{:.20f}".format(prob))
    print("Probability for NOT having a Chagas parasites:", "{:.20f}".format(1.0 - prob))


# Estimate the parameters of your Gaussian discriminat classifier
def main():
    negatives = calculate_feature_vectors("negatives")
    negative_mean = mean_for_feature_vectors(negatives)
    print("negative_mean", negative_mean)

    positives = calculate_feature_vectors("positives")
    positives_mean = mean_for_feature_vectors(positives)
    print("positives_mean", positives_mean)

    coefficient_matrix = np.zeros((6, 6))

    for pos in positives:
        defuck = np.array([pos - positives_mean])
        coefficient_matrix += defuck * np.transpose(defuck)

    for neg in negatives:
        defuck = np.array([neg - negative_mean])
        coefficient_matrix += defuck * np.transpose(defuck)

    coefficient_matrix = coefficient_matrix / (len(negatives) + len(positives))
    print("coefficient_matrix", coefficient_matrix)
    # print("det", np.linalg.det(coefficient_matrix))

    img = dataDir + "/negatives/n01.png"
    print("Negative Example:", img)
    classify_image(img, positives_mean, negative_mean, coefficient_matrix)

    img = dataDir + "/positives/p12.png"
    print("Positive Example:", img)
    classify_image(img, positives_mean, negative_mean, coefficient_matrix)


if __name__ == "__main__":
    main()
