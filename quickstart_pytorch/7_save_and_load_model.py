import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)

# ! values only, no shape

# store model
torch.save(model.state_dict(), "model_weights.pth")


# load model
model = (
    models.vgg16()
)  # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# ! values and shape

# store model
torch.save(model, "model.pth")

# load model
model = torch.load("model.pth")
