import os
import math
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Set, Dict, Tuple, Optional

from libsvm.commonutil import svm_read_problem
from libsvm.svmutil import *


# For each one of the 60 images obtain a feature vector.
def img_to_feature_vector(img: np.ndarray) -> Dict:
    red_min = np.min(img[:, :, 0])
    green_min = np.min(img[:, :, 1])
    blue_min = np.min(img[:, :, 2])

    red_mean = np.mean(img[:, :, 0])
    green_mean = np.mean(img[:, :, 1])
    blue_mean = np.mean(img[:, :, 2])

    return {
        1: red_min,
        2: green_min,
        3: blue_min,
        4: red_mean,
        5: green_mean,
        6: blue_mean,
    }


def calculate_feature_vectors(src: str) -> List[np.array]:
    vectors = []

    files = os.listdir(src)
    for file in files:
        if not file.endswith(".png"):
            continue

        path = src + "/" + file
        # print(path)

        img = skimage.io.imread(path)
        feature_vector = img_to_feature_vector(img)

        vectors += [feature_vector]

    return vectors


def main():
    positives_all = calculate_feature_vectors("positives")
    negatives_all = calculate_feature_vectors("negatives")

    test_batch_size = 1

    # Use the excludes for testing.
    test_features_pos = positives_all[-test_batch_size:]
    test_features_neg = negatives_all[-test_batch_size:]
    test_features = test_features_pos + test_features_neg

    test_labels = []
    test_labels += [+1 for _ in test_features_pos]
    test_labels += [-1 for _ in test_features_neg]

    # Compose training data

    positives = positives_all[:-test_batch_size]
    negatives = negatives_all[:-test_batch_size]

    labels = []
    labels += [+1 for _ in positives]
    labels += [-1 for _ in negatives]

    features = []
    features += positives
    features += negatives

    # print("labels", labels)
    # print("labels.len", len(labels))
    # print("features", features)
    # print("features.len", len(features))

    print("------------- svm_train -------------")
    # model = svm_train(labels, features, '-t 2 -d 2 -g 0.001 -c 0.01')
    model = svm_train(labels, features, '-t 1 -d 1')
    # model = svm_train(labels, features, '-c 4')

    print("------------- svm_predict -------------")
    p_label, p_acc, p_val = svm_predict(test_labels, test_features, model)

    print("p_label", p_label)
    print("p_acc", p_acc)
    print("p_val", p_val)

main()
