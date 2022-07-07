from typing import Literal
from numpy import Infinity
import torch
from SynsetMapper import SynsetMapper
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn


def evaluate_model(
    device: Literal["cuda", "cpu"],
    model: nn.Module,
    path_to_model_data: str,
    preprocessing_transformation,
    mapper: SynsetMapper,
):
    # load model
    checkpoint = torch.load(path_to_model_data)
    model.load_state_dict(checkpoint["model_state_dict"])

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
        plt.waitforbuttonpress(timeout=0.5)

        # normalizing transformation
        tensor = preprocessing_transformation(img)
        plt.title(
            "Normalized image: " + file_name,
        )
        plt.imshow(transforms.ToPILImage()(tensor))
        plt.waitforbuttonpress(timeout=0.5)

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
