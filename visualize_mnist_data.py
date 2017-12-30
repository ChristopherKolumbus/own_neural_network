import os
import csv

import numpy as np
from matplotlib import pyplot as plt


def main():
    path = r'..\..\pycharm_projects_data\mnist_data'
    filename = 'mnist_train_100.csv'
    with open(os.path.join(path, filename), 'r') as file_object:
        file_reader = csv.reader(file_object, delimiter=',')
        all_data = list(file_reader)
    data = all_data[3]
    image = np.array(data[1:], dtype=float).reshape((28, 28))
    plt.imshow(image, cmap='Greys', interpolation='None')
    plt.show()


if __name__ == '__main__':
    main()
