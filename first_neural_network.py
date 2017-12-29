class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3
    neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


if __name__ == '__main__':
    main()
