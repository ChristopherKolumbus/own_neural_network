import argparse

import numpy as np
from scipy import misc

from first_neural_network import NeuralNetwork, prepare_inputs


def query_neural_net(img_path):
    img_array = misc.imread(img_path, flatten=True)
    img_data = 255.0 - img_array.reshape(784)
    img_data = np.insert(img_data, 0, 0.0)
    img_data = prepare_inputs(img_data)
    neural_network = NeuralNetwork()
    neural_network.load_weights()
    result = np.argmax(neural_network.query(img_data))
    print(f'Answer from neural network: {result}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='.png file of size 28x28')
    args = parser.parse_args()
    img_path = args.img_path
    query_neural_net(img_path)


if __name__ == '__main__':
    main()
