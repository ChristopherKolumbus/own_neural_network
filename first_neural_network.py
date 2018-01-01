import csv
import os

import numpy as np
from scipy import special
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_ih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_ho = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        # self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        # self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        self.lr = learning_rate

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        output_inputs = np.dot(self.weights_ho, hidden_outputs)
        output_outputs = self.activation_function(output_inputs)
        output_errors = targets - output_outputs
        hidden_errors = np.dot(self.weights_ho.T, output_errors)
        self.weights_ho += self.lr * np.dot(output_errors * output_outputs * (1 - output_outputs), hidden_outputs.T)
        self.weights_ih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        output_inputs = np.dot(self.weights_ho, hidden_outputs)
        output_outputs = self.activation_function(output_inputs)
        return output_outputs

    @staticmethod
    def activation_function(x):
        return special.expit(x)


def visualize_number(record):
    image_array = np.array(record[1:], dtype=float).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.title(str(record[0]))
    plt.show()


def prepare_inputs(record):
    return np.array(record[1:], dtype=float) / 255 * 0.99 + 0.01


def main():
    # Specify number of input, hidden and output nodes:
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    # Learning rate of neural network:
    learning_rate = 0.3
    # Create neural network instance:
    neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # Load mnist training data from CSV file into list:
    data_folder = r'..\..\google_drive\python_data\mnist_data'
    with open(os.path.join(data_folder, 'mnist_train_100.csv'), 'r') as training_data_file:
        training_data_file_reader = csv.reader(training_data_file, delimiter=',')
        training_data_list = list(training_data_file_reader)
    # Train neural network:
    for record in training_data_list:
        # Scale and shift the inputs:
        inputs = prepare_inputs(record)
        # Create target output values (all 0.01 expect desired label which is 0.99):
        targets = np.zeros(output_nodes) + 0.01
        targets[int(record[0])] = 0.99
        neural_network.train(inputs, targets)
    # Load mnist test data from CSV file into list:
    with open(os.path.join(data_folder, 'mnist_test_10.csv'), 'r') as test_data_file:
        test_data_file_reader = csv.reader(test_data_file, delimiter=',')
        test_data_list = list(test_data_file_reader)
    # Test neural network:
    record = test_data_list[3]
    visualize_number(record)
    output_outputs = neural_network.query(prepare_inputs(record))
    print(output_outputs)


if __name__ == '__main__':
    main()
