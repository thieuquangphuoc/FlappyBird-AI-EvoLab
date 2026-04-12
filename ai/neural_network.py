import numpy as np
import random
import config

from typing import Union

class NeuralNetwork:
    def __init__(self, input_nodes=config.INPUT_NODE, hidden_nodes=config.HIDDEN_NODE, output_nodes=config.OUTPUT_NODE) -> None:
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # weight layer 1 and 2
        self.w1 = np.random.randn(self.input_nodes, self.hidden_nodes)
        self.w2 = np.random.randn(self.hidden_nodes, self.output_nodes)

        # bias for layer 1 and 2
        self.b1 = np.random.randn(1, self.hidden_nodes)
        self.b2 = np.random.randn(1, self.output_nodes)

    # sigmoid activation function, comrepssing into [0,1]
    def sigmoid(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 1 / (1 + np.exp(-x))
    
    # RelU activation function, comressing into max(0,x)
    def relu(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.maximum(0, x)

    # calculating matrix
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        # matrix 1x4 4x5 -> 1x5
        hidden = self.relu(np.dot(inputs, self.w1) + self.b1)
        
        # matrix 1x5 5x1 -> 1x1
        output = self.sigmoid(np.dot(hidden, self.w2) + self.b2)
        return np.array(output).flatten()
    
    def copy(self):
        new_brain = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)

        new_brain.w1 = np.copy(self.w1)
        new_brain.w2 = np.copy(self.w2)

        new_brain.b1 = np.copy(self.b1)
        new_brain.b2 = np.copy(self.b2)
        
        return new_brain
    
    def mutate(self, rate=config.MUTATION_RATE):
        # loop each weight in weight layer 1
        for i in range(self.w1.shape[0]):
            for j in range(self.w1.shape[1]):
                # if random num less than rate -> mutate
                if random.random() < rate:
                    self.w1[i][j] += np.random.randn() * 0.5

        # loop each weight in weight layer 2
        for i in range(self.w2.shape[0]):
            for j in range(self.w2.shape[1]):
                if random.random() < rate:
                    self.w2[i][j] += np.random.randn() * 0.5

        for i in range(self.b1.shape[1]):
            if random.random() < rate:
                self.b1[0][i] += np.random.randn() * 0.5
        for i in range(self.b2.shape[1]):
            if random.random() < rate:
                self.b2[0][i] += np.random.randn() * 0.5

    def crossover(self, partner: 'NeuralNetwork') -> 'NeuralNetwork':
        # spanw new brain
        child_brain = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)

        # create mask that has the same size of W1
        mask_w1 = np.random.rand(*self.w1.shape)

        # arithmetic crossover
        # eg: mask - 0.8 -> take 80% of father, and 20% of mother
        child_brain.w1 = mask_w1 * self.w1 + (1 - mask_w1) * partner.w1

        mask_w2 = np.random.rand(*self.w2.shape)
        child_brain.w2 = mask_w2 * self.w2 + (1 - mask_w2) * partner.w2

        mask_b1 = np.random.rand(*self.b1.shape)
        child_brain.b1 = mask_b1 * self.b1 + (1 - mask_b1) * partner.b1

        mask_b2 = np.random.rand(*self.b2.shape)
        child_brain.b2 = mask_b2 * self.b2 + (1 - mask_b2) * partner.b2
        
        return child_brain