import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),  # can use a custom callback. this converts images or ndarrays into FloatTensors, with intensities scaled from 0->1
    target_transform=Lambda(  # can use a custom callback. this converts string/indexed labels into a one-hot vector
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    ),
)
