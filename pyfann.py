import pickle
from random import normalvariate, shuffle


class Neuron:
    '''A neuron.
    
    A neuron accepts values from all neurons in the previous layer and combines them into a single value, which is passed to the next layer.

    Attributes:
        weights (list): The weights associated with all neurons in the previous layer
        bias (float): The neuron's bias
        value (float): The value of the neuron
        val_prime (float): The derivative the neuron's activation function at the neuron's value

    Methods:
        feed(sources): Combines all inputs (given by sources) to the neuron into a single value and assigns the result to the neuron's value field
    '''

    def __init__(self, weights, bias):
        '''Params:
            weights (list): A list of weights associated with the previous layer
            bias (float): The neuron's bias
        '''

        self.value = 0
        self.val_prime = 0
        self.weights = weights
        self.bias = bias

    def feed(self, sources):
        '''Takes all inputs to the current neuron, manipulates them, and assigns the result to the neuron's value parameter.

        Each input is first multiplied by its respective weight, then the results are summed, and finally the sigmoid of that sum is taken.

        Params:
            sources (list): A list of all neurons feeding into the current neuron

        Returns: None
        ''' 

        self.value = self.bias
        for i in range(len(sources)):
            self.value += self.weights[i] * sources[i].value

        if self.value < 0:
            self.value = 0

        if self.value > 0:
            self.val_prime = 1
        else:
            self.val_prime = 0


class Network:
    '''A network of neurons with dimensions specified by size.

    Attributes:
        size (list): A list specifying the dimension of the network
        neurons(list): A list of the neurons composing the network
        weights_file (str): The name of the file in which the network's weights are saved
        biases_file (str): The name of the file in which the network's biases are saved

    Methods:
        run(inputs): Takes the given inputs, passes them to the network, and assigns the network's output to the output layer
        save_neurons(): Saves all current weights and biases to weights_dict.txt and biases_dict.txt respectively
        train(inputs, outputs_expected, rate, epochs): Trains the network according to given inputs and expected outputs
    '''

    def __init__(self, size, *, weights_file = 'weights_dict.txt', biases_file = 'biases_dict.txt'):
        '''Params:
            size (list): The dimensions of the network. Should be a list where each elements represents a new layer, 
                         and the value of each element is the number of neurons in that layer
            weights_file (str, optional): File to load weights for the network from. If a file of the specified name does
                                          not exist, it is created
            biases_file (str, optional): File to load biases for the network from. If a file of the specified name does
                                          not exist, it is created
           
        Calls the function create_network after initializing variables
        '''

        self.size = size
        self.neurons = []
        self.weights_file = weights_file
        self.biases_file = biases_file
        self.__create_network()

    def __load_weights(self):
        # Attemps to load saved weights for the network from weights_file.

        # Check if file exists. If not, create it
        try:
            open(self.weights_file).close()
        except:
            open(self.weights_file, 'w').close()

        weights = []
        weights_valid = True

        # Check if file is empty
        with open(self.weights_file, 'rb') as weights_dict:
            try:
                weights = pickle.load(weights_dict)
            except:
                weights_valid = False
        
        # Check if the dimensions of the weights dictionary are the same as the network
        if len(weights) != len(self.size) - 1:
            weights_valid = False
        for layer in range(len(weights)):
            if len(weights[layer]) != self.size[layer + 1]:
                weights_valid = False
                break

        # If the dimensions of the weights and the network differ or there is no saved dictionary, 
        # a new dictionary is created with random weights
        if not weights_valid:
            weights = []
            for layer in range(len(self.size) - 1):
                weights.append([[normalvariate(0, (2 / self.size[layer]) ** 0.5) for pos_previous in range(self.size[layer])] for pos_current in range(self.size[layer + 1])])

        return weights

    def __load_biases(self):
        # Attemps to load saved biases for the network from biases_file.

        # Check if file exists. If not, create it
        try:
            open(self.biases_file).close()
        except:
            open(self.biases_file, 'w').close()

        biases = []
        biases_valid = True

        # Check if file is empty
        with open(self.biases_file, 'rb') as biases_dict:
            try:
                biases = pickle.load(biases_dict)
            except:
                biases_valid = False
        
        # Check if the dimensions of the weights dictionary are the same as the network
        if len(biases) != len(self.size) - 1:
            biases_valid = False
        for layer in range(len(biases)):
            if len(biases[layer]) != self.size[layer + 1]:
                biases_valid = False
                break

        # If the dimensions of the weights and the network differ or there is no saved dictionary, 
        # a new dictionary is created with random weights
        if not biases_valid:
            biases = []
            for layer in range(len(self.size) - 1):
                biases.append([0 for pos in range(self.size[layer + 1])])

        return biases

    def __create_network(self):
        # Fills network's neurons list with new neuron objects according to the dimensions specified by size.

        weights = self.__load_weights()
        biases = self.__load_biases()

        # Fills neurons list according to dimensions specified by size
        # All neurons in the input layer are assigned weights and biases of None
        self.neurons.append([Neuron(None, None) for pos in range(self.size[0])])
        for layer in range(1, len(self.size)):
            self.neurons.append([Neuron(weights[layer - 1][pos], biases[layer - 1][pos]) for pos in range(self.size[layer])])

    def __del_loss(self, outputs_real, train_pos, mode):
        # Calculates the partial derivative of the loss function with respect to the given weight or bias.

        train_layer = train_pos[0]
        train_neuron = self.neurons[train_layer][train_pos[1]]
        
        # Implementation of the chain rule to take the derivative of each neuron's value
        total = [1]
        if train_layer != len(self.neurons) - 1: # If training the output layer, next part is skipped
            total = [neuron.weights[train_pos[1]] * neuron.val_prime for neuron in self.neurons[train_layer + 1]]
            if train_layer != len(self.neurons) - 2: # If training the layer just before the output, next part is skipped
                for layer in range(train_layer + 2, len(self.neurons)):
                    total_next = []
                    for neuron in self.neurons[layer]:
                        sum =  0
                        for weight in range(len(neuron.weights)):
                            sum += neuron.weights[weight] * total[weight]
                        total_next.append(sum * neuron.val_prime)
                    total = total_next

        if mode == 0: # If training a weight
            total = [i * self.neurons[train_layer - 1][train_pos[2]].value * train_neuron.val_prime for i in total]
        elif mode == 1: # If training a bias
             total = [i * train_neuron.val_prime for i in total]

        del_loss = 0
        if train_layer == len(self.neurons) - 1: # If training the output layer
            output_real = outputs_real[train_pos[1]]
            output_predicted = self.neurons[-1][train_pos[1]].value
            del_loss = -2 * (output_real - output_predicted) * total[0]
        else: # Sum the derivatives of all output neurons
            for output in range(len(total)):
                output_predicted = self.neurons[-1][output].value
                del_loss += -2 * (outputs_real[output] - output_predicted) * total[output]
            
        return del_loss

    def save_neurons(self):
        '''Saves all current weights to the file specified by weights_file and all current biases to the file specified by biases_file.
        
        Returns: None
        '''

        # Read weights and biases from network
        weights = [[neuron.weights for neuron in self.neurons[layer]] for layer in range(1, len(self.neurons))]
        biases = [[neuron.bias for neuron in self.neurons[layer]] for layer in range(1, len(self.neurons))]

        # Save weights and biases to respective files
        with open(self.weights_file, 'wb') as weights_dict, open(self.biases_file, 'wb') as biases_dict:
            pickle.dump(weights, weights_dict)
            pickle.dump(biases, biases_dict)

    def run(self, inputs):
        '''Takes the given inputs and propogates them through the network.

        Params:
            inputs (list): A list of inputs for the network. The size of inputs must be the same as the size of the input layer

        Returns: None
        '''

        # Checks to see if the right number of inputs were given
        if len(inputs) != len(self.neurons[0]):
            return print('Invalid input')

        # Assigns given inputs to the value of the neurons in the input layer,
        # then runs the feed function for the remaining neurons
        for i in range(len(inputs)):
            self.neurons[0][i].value = inputs[i]
        for layer in range(1, len(self.neurons)):
            for pos in range(len(self.neurons[layer])):
                self.neurons[layer][pos].feed(self.neurons[layer - 1])

    def train(self, inputs, outputs_expected, rate, epochs):
        '''Trains the network given inputs and expected outputs.

        Params:
            inputs (list): All inputs to be passed to the network. Must be in the form of a 2-dimensional list
            outputs_expected (list): All the expected outputs given the inputs. The index of each element in outputs_expected should be 
                                     the same as the index of the corresponding input
            rate (float): The desired training rate for the network
            epochs (int): The desired number of training iterations for given inputs

        Returns: None

        Effects:
            Automatically saves all weights and biases when training is finished.
        '''

        for _ in range(epochs):
            # Shuffle inputs and ouputs
            shuffled = list(zip(inputs, outputs_expected))
            shuffle(shuffled)
            inputs, outputs_expected = zip(*shuffled)

            for input_, outputs in zip(inputs, outputs_expected):
                self.run(input_)
                # Iterate over every neuron in the network (excluding input layer)
                for layer in range(1, len(self.neurons)):
                    for pos in range(len(self.neurons[layer])):
                        # Adjust bias and all weights by their respective gradient times the training rate
                        for weight in range(len(self.neurons[layer][pos].weights)):
                            self.neurons[layer][pos].weights[weight] -= rate * self.__del_loss(outputs, [layer, pos, weight], 0)
                        self.neurons[layer][pos].bias -= rate * self.__del_loss(outputs, [layer, pos, weight], 1)

        # Some pretty output indicating the training progress    
            if _ % (epochs / 100) == 0:
                percent_done = int(100 * _ / epochs)
                print('. ' * int(percent_done / 10), ' ' * (18 - 2 * int(percent_done / 10)), '{:3d}%'.format(percent_done), end='\r')
        print('. . . . . . . . . . Done\a')
        
        self.save_neurons()
