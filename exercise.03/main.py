# -*- coding: utf-8 -*-

import os
import math
import skimage.io
import matplotlib.pyplot as plt
import numpy as np

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


def calculate_feature_vectors(src: str) -> [np.array]:

    vectors = []

    source = dataDir + "/" + src
    files = os.listdir(source)
    for file in files:
        if not file.endswith(".png"):
            continue

        path = source + "/" + file
        #print(path)

        img = skimage.io.imread(path)
        feature_vector = img_to_feature_vector(img)

        vectors += [feature_vector]

    return vectors


def main():

    negatives = calculate_feature_vectors("negatives")
    negative_mean = mean_for_feature_vectors(negatives)
    print("negative_mean", negative_mean)

    positives = calculate_feature_vectors("positives")
    positives_mean = mean_for_feature_vectors(positives)
    print("positives_mean", positives_mean)

    coefficient_matrix = np.zeros((6, 6))

    for pos in positives:
        coefficient_matrix += pos * pos


if __name__ == "__main__":
    main()
