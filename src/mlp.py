import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward_pass(self, inputs, targets, outputs):
        error = targets - outputs
        d_error = error * self.sigmoid_derivative(outputs)
        hidden_error = d_error.dot(self.weights_hidden_output.T)
        d_hidden_error = hidden_error * self.sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(d_error) * self.learning_rate
        self.bias_output += np.sum(d_error, axis=0) * self.learning_rate
        self.weights_input_hidden += inputs.T.dot(d_hidden_error) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden_error, axis=0) * self.learning_rate

    def train(self, inputs, targets, epochs=10000):
        for _ in range(epochs):
            outputs = self.forward_pass(inputs)
            self.backward_pass(inputs, targets, outputs)

if __name__ == "__main__":
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
    mlp.train(inputs, targets)
    for input_data in inputs:
        output = mlp.forward_pass(input_data)
        print(f"Input: {input_data} -> Output: {output}")
