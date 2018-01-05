import numpy as np
from scipy import misc

from first_neural_network import NeuralNetwork, prepare_inputs

img_array = misc.imread('number_2.png', flatten=True)
img_data = 255.0 - img_array.reshape(784)
image_data = prepare_inputs(img_data)
neural_network = NeuralNetwork()
neural_network.load_weights()
result = np.argmax(neural_network.query(img_data))
print(f'Answer from neural network: {result}')