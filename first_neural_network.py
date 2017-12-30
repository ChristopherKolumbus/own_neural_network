import numpy as np
from scipy import special


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
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


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3
    neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(neural_network.query([1.0, 0.5, -1.5]))


if __name__ == '__main__':
    main()
