import torch
import torch.nn as nn
import torch.optim as optim

class MLP_PyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_PyTorch, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

if __name__ == "__main__":
    inputs = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = torch.Tensor([[0], [1], [1], [0]])

    model = MLP_PyTorch(input_size=2, hidden_size=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(10000):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for input_data in inputs:
            output = model(input_data)
            print(f"Input: {input_data.numpy()} -> Output: {output.numpy()}")
