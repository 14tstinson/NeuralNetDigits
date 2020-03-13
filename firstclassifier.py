import numpy as np
import random

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #creates array of biases for each layer of neurons (excluding input)
        #randn gives the array shape (y, 1) so bias for each neuron
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:]]

    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
        #implemention sigmoid function which squishes input between 0 and 1, hence sigmoid neurons

    def feedforward(self, a):
    #this is what each neuron does to it's input
    for b, w, in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a)+b)

    return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        #Stochastic gradient descent, this is the function where 'learning' happens
        #stochastic = splitting data into smaller chunks then doing gradient descent with gradient
        #calculated from the smaller batch
        #eta is learning rate
        if test_data:
            n_test = len(test_data)
            n = len(training_data)
        for j in range(epochs): #1 epoch is going through all of the test data once
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
        if test_data:
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
        else:
            print("Epoch %d complete", j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nb+dnw for nw dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.baises, nabla_b)};

