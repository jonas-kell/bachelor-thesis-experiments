import torchvision.transforms as transforms
import torch

resize_normalize = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        # these are the image net means ->
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.ToPILImage(),
    ],
)

resize_normalize_to_tensor = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        # these are the image net means ->
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ],
)

normalize_inverse = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)
