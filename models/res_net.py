import torch


def res_net_pretrained():
    return torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)


def res_net():
    return torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
