import math
import random
import numpy as np
import matplotlib.pyplot as plt


def normal_distribution(x: float, mean: float, varianz: float) -> float:
    pi = varianz * math.sqrt(2 * math.pi)
    expo = math.exp(-0.5 * ((x - mean) / varianz) ** 2)

    return expo / pi


def main():
    np.random.seed(19980528)
    random.seed(19980528)

    # Number of actions
    n = 10

    # Epsilon
    random_factor = 0.1

    # Init stuff
    iterations = 1000
    q_init_means = np.random.normal(0, 1, n)
    q_init_means.sort()

    print("q_init_means", q_init_means)

    # Q(a) <- 0
    q_a = np.zeros((n, 1))

    # N(a) <- 0
    n_a = np.zeros((n, 1))

    plot = np.zeros((n, iterations))

    for x in range(iterations):

        action = -1

        if random.random() > random_factor:

            best_reward = -1
            for inx in range(n):

                if best_reward < q_a[inx]:
                    action = inx
                    best_reward = q_a[inx]
        else:
            action = random.randint(0, n - 1)

        n_a[action] += 1

        reward = np.random.normal(q_init_means[action], 1, 1)[0]
        q_a[action] += (reward - q_a[action]) / n_a[action]

        for inx in range(n):
            plot[inx][x] = q_a[inx]

    print("q_a", q_a)
    plt.figure(0)
    for inx in range(n):
        plt.plot(plot[ inx ])

    plt.show()


if __name__ == '__main__':
    main()
