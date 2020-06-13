import os
import math
import numpy as np

data = np.loadtxt("dataCircle.txt")
print("data", data)


def weak_classifier(feature_vetor: np.array) -> int:
    print("feature_vetor", feature_vetor)

    return 1


def weight_based_on_error(error: float) -> float:
    return 0.5 * math.log((1 - error) / error)


def error() -> float:
    count = 0

    for point in data:
        if weak_classifier(point[:2]) != point[2]:
            count += 1

    return count / len(data)


err = error()
print("err", err)
