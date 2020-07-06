import math
import random
import numpy as np
import matplotlib.pyplot as plt


def normal_distribution(x: float, mean: float, varianz: float) -> float:
    pi = varianz * math.sqrt(2 * math.pi)
    expo = math.exp(-0.5 * ((x - mean) / varianz) ** 2)

    return expo / pi


def bandit_algorithm(n: int, q_means: np.array, randomization: float) -> np.array:
    # Q(a) <- 0
    q_a = np.zeros((n, 1))

    # N(a) <- 0
    n_a = np.zeros((n, 1))

    iterations = 1000
    plot = np.zeros((n, iterations))

    for x in range(iterations):

        action = -1

        if random.random() > randomization:

            best_reward = -1
            for inx in range(n):

                if best_reward < q_a[inx]:
                    action = inx
                    best_reward = q_a[inx]
        else:
            action = random.randint(0, n - 1)

        n_a[action] += 1

        reward = np.random.normal(q_means[action], 1, 1)[0]
        q_a[action] += (reward - q_a[action]) / n_a[action]

        for inx in range(n):
            plot[inx][x] = q_a[inx]

    return q_a


def main():
    np.random.seed(19980528)
    random.seed(19980528)

    # Number of actions
    n = 10

    # Epsilon
    randomization = 0.1

    # Init stuff
    q_means = np.random.normal(0, 1, n)
    q_means.sort()

    print("q_means", q_means)

    rounds = 2000
    q_average = np.zeros((n, rounds))

    for inx in range(rounds):
        q_average[:, inx] = bandit_algorithm(n, q_means, randomization).reshape((10))

        if inx % 200 == 0:
            print("t =", inx)

    print("q_average", np.mean(q_average, axis=1))


if __name__ == '__main__':
    main()
