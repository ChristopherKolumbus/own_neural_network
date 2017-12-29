import numpy as np
from matplotlib import pyplot as plt


def generate_insect_data(insect_count):
    insects = np.random.random((insect_count, 2))
    insect_types = np.random.random_integers(0, 1, size=(insects.shape[0], 1))
    for insect, insect_type in zip(insects, insect_types):
        if insect_type == 0:
            insect[0] += 1
            insect[1] += 3
        else:
            insect[0] += 3
            insect[1] += 1
    return insects, insect_types


def plot_insects(insects, insect_types, slope):
    for insect, insect_type in zip(insects, insect_types):
        if insect_type == 0:
            plt.plot(insect[0], insect[1], 'go')
        else:
            plt.plot(insect[0], insect[1], 'ro')
    x = np.arange(0, 10, 1)
    y = x * slope
    plt.plot(x, y)
    plt.show()


def classify_insects(insects, insect_types):
    slope = np.random.random()
    learning_rate = .5
    for insect, insect_type in zip(insects, insect_types):
        if insect_type == 0:
            target = insect[0] - .1
        else:
            target = insect[0] + .1
        error = target - (insect[0] * slope)
        delta_slope = error / insect[0]
        slope += delta_slope * learning_rate
    return slope


def main():
    insects, insect_types = generate_insect_data(10)
    slope = classify_insects(insects, insect_types)
    plot_insects(insects, insect_types, slope)


if __name__ == '__main__':
    main()
