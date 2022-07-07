import torch
import torch.nn as nn


class ResNetAdapter(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()

        self.pretrained = pretrained

        self.res_net = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=pretrained
        )

        self.classification_mapper = nn.Linear(1000, 100)

    def forward(self, x):

        if self.pretrained:
            with torch.no_grad():  # do not allow res net pretraining weights to be modified
                x = self.res_net(x)
        else:
            x = self.res_net(x)  # gives 1000x1 classification output

        x = self.classification_mapper(x)

        return x


def res_net():
    return ResNetAdapter(pretrained=False)


def res_net_pretrained():
    return ResNetAdapter(pretrained=True)
