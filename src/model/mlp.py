
import numpy as np

import util.loss_functions as loss_functions
from util.loss_functions import CrossEntropyError
from util.loss_functions import BinaryCrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        #self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.output = []
        self.target = 0
        
        if loss == 'bce':
            self.loss = loss_functions.BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = loss_functions.SumSquaredError()
        elif loss == 'mse':
            self.loss = loss_functions.MeanSquaredError()
        elif loss == 'different':
            self.loss = loss_functions.DifferentError()
        elif loss == 'absolute':
            self.loss = loss_functions.AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        #self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                           axis=1)

        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                             axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

        print(np.shape(self.trainingSet.input))
        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10, 
                           None, outputActivation, True))




    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        out = inp
        """
        for layer in self.layers:
            print(np.shape(out))
            out = np.append([1.0], layer.forward(out)) #np.append([1.0], out))
        self.output = out[1:]
        return out
        """

        for layer in self.layers:
            print(np.shape(out))
            out = layer.forward(out) #np.append([1.0], out))
        self.output = out
        return out

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return self.loss.calculateError(target, self._feed_forward(self.output))

    
    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """

        deltas = self.loss.calculateDerivative(self.output, self.target)
        weights = np.eye(np.size(deltas))

        for layer in np.flip(self.layers, 0):
            deltas = layer.computeDerivative(deltas, weights)
            weights = np.transpose(layer.weights)
            layer.updateWeights(self.learningRate)


    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            for (x, t) in zip(self.trainingSet.input, self.trainingSet.label):
                print(np.shape(x))
                self.target = t
                self._feed_forward(x)
                self._update_weights(self.learningRate)


    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        out = self._feed_forward(test_instance)
        max_index = np.argmax(out)
        output = np.zeros(np.shape(out))
        output[max_index]=1
        return output

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
