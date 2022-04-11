import numpy as np
import time

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

from MinstNeuralNetwork.tools.activation_functions import sigmoid, softmax, softsign
from MinstNeuralNetwork.tools.other import create_solution
from MinstNeuralNetwork.tools.image_processing import image_process

# Load data set from keras.minst
(x_train, y_train), (x_val, y_val) = mnist.load_data()

# convert to one-hot vector
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# image dimensions (assumed square)
image_size = x_train.shape[1]
input_size = image_size * image_size
# resize and normalize
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_val = np.reshape(x_val, [-1, input_size])
x_val = x_val.astype('float32') / 255

accuracy_list = []
epoch_list = [i for i in range(1, 6)]


class DeepNeuralNetwork:
    def __init__(self, sizes, epochs=5, l_rate=0.1):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        # save params
        self.params = self.initialization()

    def initialization(self):
        # chose the struct of DNN ( Number of nodes in each layer )
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]

        params = {
            'W1': np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2': np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3': np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = softsign(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = softsign(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        params = self.params
        change_w = {}

        # calculate w3
        error = 2 * (output - y_train) / output.shape[0] * softmax(params['Z3'])
        change_w['W3'] = np.outer(error, params['A2'])

        # calculate w2
        error = np.dot(params['W3'].T, error) * softsign(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # calculate w1
        error = np.dot(params['W2'].T, error) * softsign(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    # Compute the Stochastic Gradient Descent
    def stochastic_gradient(self, weight_changes):
        for key, value in weight_changes.items():
            self.params[key] -= self.l_rate * value

    # Compute Momentum
    def momentum(self, changes_to_w):
        i = 1
        eta = 0.9
        previousvalue = {'W1': None, 'W2': None, 'W3': None}
        for key, value in changes_to_w.items():
            if i == 1:
                self.params[key] -= self.l_rate * value
                if previousvalue['W3'] is not None:
                    self.params[key] += eta * previousvalue['W3']
                previousvalue['W3'] = self.params[key]
            if i == 2:
                self.params[key] -= self.l_rate * value
                if previousvalue['W2'] is not None:
                    self.params[key] += eta * previousvalue['W2']
                previousvalue['W2'] = self.params[key]
            if i == 3:
                self.params[key] -= self.l_rate * value
                if previousvalue['W1'] is not None:
                    self.params[key] += eta * previousvalue['W1']
                previousvalue['W1'] = self.params[key]
            i += 1

            # Compute Accuracy

    def accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)

    # Training Loop
    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        # first loop for epochs, number of time we run through all the db
        for i in range(self.epochs):
            # second loop for each data in db
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                weight_change = self.backward_pass(y, output)
                self.stochastic_gradient(weight_change)

            accuracy = self.accuracy(x_val, y_val)
            accuracy_list.append(accuracy)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                i + 1, time.time() - start_time, accuracy * 100))

    # Allow to use a trained DNN on a chosen image
    def test_on_personal_image(self, path, i):
        start = time.time()
        x_train_image = image_process(path)
        output = self.forward_pass(x_train_image)
        weight_change = self.backward_pass(create_solution(i), output)
        self.stochastic_gradient(weight_change)
        output = self.forward_pass(x_train_image)
        print(output)
        print("for input :", i, "  output is :", np.argmax(output))
        print("time : ", time.time()-start)

    # Allow to use a trained DNN on a chosen image and process this image several times
    def complete_image_test(self, path, i, epochs=10):
        x_train_image = image_process(path)
        output = []
        somme = [i for i in range(1, epochs + 1)]
        for i in range(epochs):
            output.append((self.forward_pass(x_train_image)))
        for i in range(epochs):
            for j in range(len(output[0])):
                somme[i] += output[i][j]
        result = [i / epochs for i in output]
        print("for input :", i, "  output is :", result)
        print(result)
