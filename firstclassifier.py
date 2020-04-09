import numpy as np
import random

class Network(object):
    def __init__(self, sizes, *cost_type):
        
        self.cost_type = cost_type
        self.num_layers = len(sizes)
        self.sizes = sizes
        #sizes is number of neurons in each layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #creates array of biases for each layer of neurons (excluding input)
        #randn gives the array shape (y, 1) so bias for each neuron
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        #generates random weights for each input to each node, x is one behind y so does weights for         #right ammount of inputs

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
        #implemention sigmoid function which squishes input between 0 and 1, hence sigmoid neurons

    def feedforward(self, a):
    #this is what each neuron does to it's input
    #a is inputs to neuron then returned as outputs
        for b, w, in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        #Stochastic gradient descent, this is the function where 'learning' happens
        #stochastic = splitting data into smaller chunks then doing gradient descent with gradient
        #calculated from the smaller batch
        #eta is learning rate

        training_data = list(training_data)
        n = len(training_data)

        if test_data:                        
            test_data = list(test_data)
            n_test = len(test_data)
        for j in range(epochs): #1 epoch is going through all of the test data once
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                #update the weights and biases using back propogation
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                #printing information on which epoch is finished and the sucess rate
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch %d complete", j)

    def update_mini_batch(self, mini_batch, eta):
        #this function updates the weights and the biases by the ammount calculated using the 
        #mini batch data (This is the gradient descent)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #x is input data, y is correct answer
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #currently these list loops are the same as setting nabla to delta nabla
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        #Backpropagation is the algorithm used to calculate the partial derivatives with respect to each weight and bias
        #in the network. Having these values lets us perform gradient descent in order to improve the network 'learn'.
        #Backpropagation is so good because it computes all of the partial derivatives needed with the processing equivalent
        #to going through the network once.
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #first input to get the network going
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            
            z = np.dot(w, activation)+b
            zs.append(z)
            #this loop is putting the input through the whole network as we currently have it (the feedforward)
            #saves results of all calculations in zs and activation(input for next layer) which is just sigmoid of z
            #these intermediate results are used to calculate required derivatives
            #This is great as dont need any more processing, just need to store intermediate results
            activation = self.sigmoid(z)
            activations.append(activation)
        
        #calculating delta which is our error
        #error in this sense is a list of partial derivatives of cost with respect to z of the l-th layer
        delta = self.quad_cost_derivative(activations[-1], y) 
        if cost_type == "quadratic":
            delta *= self.sigmoid_prime(zs[-1])
            
        #implementing backpropagation equation to calculate error 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            #this is the backpropagation of the error
            #go through the layers backwards to calculate partial derivatives of cost function with respect to each
            #weight and bias and return these (nabla_b, nabla_w) to be used for gradient descent
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            #this is just one of the four backpropagation equation being implemented with np operations
            #transpose just 'flips' the dimensions/axes of the array so the multiplication works (dot product)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #delta is array of errors for each neuron in last -l layer
            nabla_b[-l] = delta
            #implementing backpropagation equation
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            #implementing last backpropagation quation
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        #argmax find the output that is the most likely
        #argmax returns index of max value
        #the index is the digit
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x==y) for x, y in test_results)#
        #this returns how many of the predictions are right

    def cross_entropy_cost_deriviative(self, output_activations, y):
        return
    def quad_cost_derivative(self, output_activations, y):
        #this calculation for the partial derivative of quadratic (MSE) cost function with respect to an activation
        #This is so simple due to some chain rule terms cancelling
        return output_activations-y

    def sigmoid_prime(self, z):
        #derivative of sigmoid function
        return self.sigmoid(z)*(1-self.sigmoid(z))
