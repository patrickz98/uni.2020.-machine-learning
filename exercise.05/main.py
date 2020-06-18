import os
import math
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("dataCircle.txt")
print("data", data)


def weak_classifier(classifier: (int, int), feature_vector: np.array) -> int:
    if classifier[1] > feature_vector[classifier[0]]:
        return -1
    else:
        return 1


def weight_based_on_error(error: float) -> float:
    return 0.5 * math.log((1 - error) / error)


def error(classifier: (int, int)) -> float:
    count = 0

    for point in data:
        if weak_classifier(classifier, point) != point[2]:
            count += 1

    return count / len(data)


def show_classifier(classifier: (int, int)):
    xs_pos = []
    ys_pos = []

    xs_neg = []
    ys_neg = []

    for dataPoint in data:
        if weak_classifier(classifier, dataPoint) == -1:
            continue

        if dataPoint[2] == 1:
            xs_pos += [dataPoint[0]]
            ys_pos += [dataPoint[1]]
        else:
            xs_neg += [dataPoint[0]]
            ys_neg += [dataPoint[1]]

    plt.figure(0)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.scatter(xs_pos, ys_pos)
    plt.scatter(xs_neg, ys_neg)

    if classifier[0] == 0:
        plt.axvline(classifier[1])
    else:
        plt.axhline(classifier[1])

    plt.show()


def main():
    classifier = []

    # dimension: 0 --> Vertical, 1 --> horizontal
    for dimension in [0, 1]:
        for separation_line in range(-10, 10, 2):
            classifier += [(dimension, separation_line)]

    show_classifier(classifier[17])
    print("error", error(classifier[17]))

    # plt.figure(0)
    # for cl in classifier:
    #     if cl[0] == 1:
    #         plt.axvline(cl[1])
    #     else:
    #         plt.axhline(cl[1])
    #
    # plt.show()


main()
