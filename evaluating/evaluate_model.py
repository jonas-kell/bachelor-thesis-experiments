from typing import Literal
from numpy import Infinity
import torch
from SynsetMapper import SynsetMapper
from transformations import resize_normalize
import sys
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# the used model definition must be available in a file in the following folder


# ML-Perceptron: the file must be named 'model.py' and the network class 'NeuralNetwork'
sys.path.append(
    "/media/jonas/69B577D0C4C25263/MLData/tensorboard/ML-Perceptron-RandSize/"
)

# Transformer-tests: can be importet, if the path to DINO-CLASSIFIER/vision_transformer.py is set
sys.path.append("/media/jonas/69B577D0C4C25263/MLData/tensorboard/DINO-CLASSIFIER/")


def evaluate_model(
    device: Literal["cuda", "cpu"],
    path_to_model_data: str,
    preprocessing_transformation,
    mapper: SynsetMapper,
):
    # load model
    model = torch.load(path_to_model_data)
    model.to(device)

    # image files to process
    file_names = os.listdir("./evaluating/data_dropin_folder")
    file_names.remove(".gitignore")

    # loop over files
    for file_name in file_names:
        # load
        file_path = os.path.join(
            os.path.abspath("./evaluating/data_dropin_folder"), file_name
        )
        img = Image.open(file_path)

        # display
        plt.title(
            "Original image: " + file_name,
        )
        plt.imshow(img)
        plt.waitforbuttonpress()

        # normalizing transformation
        tensor = preprocessing_transformation(img)
        plt.title(
            "Normalized image: " + file_name,
        )
        plt.imshow(transforms.ToPILImage()(tensor))
        plt.waitforbuttonpress()

        # move to device
        tensor = tensor[None, :]
        tensor = tensor.to(device)

        # predictions
        with torch.no_grad():
            pred = model(tensor)

            print("Synset choices for " + file_name)

            for i in range(5):
                # chose predicted class
                index = torch.argmax(pred[0]).item()

                # print result
                print(
                    "Chosen class for priority #"
                    + str(i + 1)
                    + ": "
                    + str(index)
                    + f"({(pred[0][index].item()):>0.3f})"
                    + " -> "
                    + mapper.synset_id_from_vector_index(index)
                    + " : "
                    + mapper.description_from_vector_index(index)
                )

                # remove index for second sampling
                pred[0][index] = -Infinity
