from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(224 * 224 * 3, 512 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2, 512 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2, 512 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2, 100),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
