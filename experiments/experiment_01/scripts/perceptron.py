import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(input_size + 1)  # +1 pro bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # bias + w*x
        return self.activation(summation)

    def train(self, training_data, labels):
        errors = []
        for epoch in range(self.epochs):
            total_error = 0
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error
                total_error += abs(error)
            errors.append(total_error)
        return errors

    def test(self, testing_data, labels):
        correct = 0
        for inputs, label in zip(testing_data, labels):
            if self.predict(inputs) == label:
                correct += 1
        return correct / len(labels)
