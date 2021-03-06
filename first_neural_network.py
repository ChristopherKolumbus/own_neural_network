import csv
import os
import time
import shelve

import numpy as np
from scipy import special
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self, input_nodes=10, hidden_nodes=10, output_nodes=10, learning_rate=.1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_ih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_ho = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
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

    def save_weights(self):
        with shelve.open('neural_network_weights') as weights_shelve:
            weights_shelve['weights_input_hidden'] = self.weights_ih
            weights_shelve['weights_hidden_output'] = self.weights_ho

    def load_weights(self):
        with shelve.open('neural_network_weights') as weights_shelve:
            self.weights_ih = weights_shelve['weights_input_hidden']
            self.weights_ho = weights_shelve['weights_hidden_output']


def visualize_number(record):
    image_array = np.array(record[1:], dtype=float).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.title(str(record[0]))
    plt.show()


def prepare_inputs(record):
    return np.array(record[1:], dtype=float) / 255 * 0.99 + 0.01


def train_neural_network(neural_network, output_nodes, epochs, training_data_path):
    for epoch in range(epochs):
        print(f'Training neural network (epoch {epoch + 1}/{epochs})...')
        t0 = time.time()
        # Load mnist training data from CSV file:
        with open(training_data_path, 'r') as training_data_file:
            training_data_file_reader = csv.reader(training_data_file, delimiter=',')
            for record in training_data_file_reader:
                # Scale and shift the inputs:
                inputs = prepare_inputs(record)
                # Create target output values (all 0.01 expect desired label which is 0.99):
                targets = np.zeros(output_nodes) + 0.01
                targets[int(record[0])] = 0.99
                # Train neural network:
                neural_network.train(inputs, targets)
            t1 = time.time()
            total = t1 - t0
            print(f'Done! (epoch {epoch + 1}/{epochs}; {total}s)')


def test_neural_network(neural_network, test_data_path):
    print('Testing neural network...')
    t0 = time.time()
    # Load mnist test data from CSV file:
    with open(test_data_path, 'r') as test_data_file:
        test_data_file_reader = csv.reader(test_data_file, delimiter=',')
        score_card = []
        for record in test_data_file_reader:
            correct_label = int(record[0])
            inputs = prepare_inputs(record)
            label = np.argmax(neural_network.query(inputs))
            if label == correct_label:
                score_card.append(1)
            else:
                score_card.append(0)
        score_card_array = np.array(score_card)
        performance = score_card_array.sum() / score_card_array.size
        t1 = time.time()
        total = t1 - t0
        print(f'Done! ({total}s)\nPerformance = {performance}')


def main():
    # Specify number of input, hidden and output nodes:
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    # Learning rate of neural network:
    learning_rate = 0.1
    # Number of epochs:
    epochs = 5
    # Create neural network instance:
    neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # Train neural network:
    data_folder = r'..\..\google_drive\python_data\mnist_data'
    train_neural_network(neural_network, output_nodes, epochs, os.path.join(data_folder, 'mnist_train.csv'))
    # Save weights:
    neural_network.save_weights()
    # Test neural network:
    test_neural_network(neural_network, os.path.join(data_folder, 'mnist_test.csv'))


if __name__ == '__main__':
    main()
