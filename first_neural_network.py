import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.weights_hidden_output = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

    def train(self):
        pass

    def query(self):
        pass


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3
    neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


if __name__ == '__main__':
    main()
