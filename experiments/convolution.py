"""
Experiments following on the article https://learnopencv.com/fully-convolutional-image-classification-on-arbitrary-sized-image/

repo: https://github.com/spmallick/learnopencv/tree/master/PyTorch-Fully-Convolutional-Image-Classification
"""


import torch
from torch import nn


#                         C, H, W
test_tensor = torch.ones((2, 3, 4))
print(test_tensor)

# Fully connected layer
fcl = nn.Linear(in_features=2 * 3 * 4, out_features=5, bias=False)
print(fcl(test_tensor.reshape((1, 2 * 3 * 4))))  # -> 1, 5

# convolutions
conv_1x1 = nn.Conv2d(in_channels=2, out_channels=5, kernel_size=1, bias=False)
print(conv_1x1(test_tensor))  # -> 5, 3, 4

conv_3x4 = nn.Conv2d(in_channels=2, out_channels=5, kernel_size=(3, 4), bias=False)
conv_3x4.weight.data.copy_((fcl.weight.data.view(5, 2, 3, 4)))
print(conv_3x4(test_tensor))  # -> 5, 1, 1

conv_1x1_code = nn.Conv2d(
    in_channels=2 * 3 * 4, out_channels=5, kernel_size=1, bias=False
)
conv_1x1_code.weight.data.copy_((fcl.weight.data.view(5, 2 * 3 * 4, 1, 1)))
print(conv_1x1_code(test_tensor.reshape(2 * 3 * 4, 1, 1)))  # -> 5, 1, 1
