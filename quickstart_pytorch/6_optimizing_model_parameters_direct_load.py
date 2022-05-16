import time

start_time = time.time()

# ! Code from previous examples 2 and 4, expanded to do gpu

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512 * 8),
            nn.ReLU(),
            nn.Linear(512 * 8, 512 * 8),
            nn.ReLU(),
            nn.Linear(512 * 8, 512 * 4),
            nn.ReLU(),
            nn.Linear(512 * 4, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

# ! hyperparameters

learning_rate = 1e-3
epochs = 30  # epoch: train loop + validation/test
batch_size = 64

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ! push datasets to gpu as a whole (test)
def push_to_cuda(dataloader):
    dataset_pushed = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for batch, (X, y) in enumerate(dataloader):
        dataset_pushed.append((batch, (X.to(device), y.to(device))))

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"preloading to gpu: [{current:>5d}/{size:>5d}]")

    return type(
        "",
        (object,),
        {"dataset": dataset_pushed, "size": size, "num_batches": num_batches},
    )()


# ! def loops


def train_loop(dataloader, model, loss_fn, optimizer):
    for batch, (X, y) in dataloader.dataset:
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{dataloader.size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    test_loss, correct = 0, 0

    with torch.no_grad():
        for _, (X, y) in dataloader.dataset:
            # test
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= dataloader.num_batches
    correct /= dataloader.size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# ! train it like it's hot

train_dataloader = push_to_cuda(DataLoader(training_data, batch_size=batch_size))
test_dataloader = push_to_cuda(DataLoader(test_data, batch_size=batch_size))

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

print("--- %s seconds ---" % (time.time() - start_time))
