import inspect
import sys
import os
from typing import Literal
import torch
from torch import nn

sys.path.append(os.path.abspath("./managing_imagenet_data"))
sys.path.append(os.path.abspath("./slack"))
sys.path.append(os.path.abspath("./train"))
sys.path.append(os.path.abspath("./models"))

from train.training_script import train_model
from managing_imagenet_data.pre_process_data import (
    transform_validation_data,
    transform_training_data,
    show_image_from_transformed_stored_tensor,
)
from transformations import resize_normalize
from PathAndFolderConstants import PathAndFolderConstants
from SynsetMapper import SynsetMapper
from evaluating.evaluate_model import evaluate_model

constants = PathAndFolderConstants(
    path_to_imagenet_data_folder="/media/jonas/69B577D0C4C25263/MLData/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC",
    path_to_folder_for_transformed_data="/media/jonas/69B577D0C4C25263/MLData/transformed",
    path_to_validation_solution_file="/media/jonas/69B577D0C4C25263/MLData/imagenet-object-localization-challenge/LOC_val_solution.csv",
    path_to_tensorboard_log_folder="/media/jonas/69B577D0C4C25263/MLData/tensorboard",
)
mapper = SynsetMapper(constants)

from models.vision_transformer import vit_custom


def prepare_data(
    constants: PathAndFolderConstants,
    mapper: SynsetMapper,
):
    transformation = resize_normalize

    transform_training_data(transformation, constants, mapper)
    transform_validation_data(transformation, constants, mapper)


def train(
    device: Literal["cuda", "cpu"],
    network: nn.Module,
    constants: PathAndFolderConstants,
    mapper: SynsetMapper,
    **kwargs,
):
    train_model(device, network, constants, mapper, **kwargs)


def evaluate(
    device: Literal["cuda", "cpu"],
    path_to_model_file: str,
    mapper: SynsetMapper,
):

    print("Using model: " + path_to_model_file + " to evaluate images")

    evaluate_model(device, path_to_model_file, resize_normalize, mapper)


if __name__ == "__main__":
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    type_of_operation = "" if len(sys.argv) == 1 else sys.argv[1]

    if type_of_operation == "prep":
        print("Prepare the data")

        prepare_data(constants, mapper)

    elif type_of_operation == "train":
        print("Train model")

        additional_parameters = [] if len(sys.argv) < 3 else sys.argv[2:]

        supported_args = inspect.getfullargspec(train_model).args
        args_to_pass = {}

        for param_string in additional_parameters:
            split = param_string.split("=", 1)

            if len(split) == 2 and split[0] in supported_args:
                print(
                    "Use additional parameter: "
                    + str(split[0])
                    + " with value "
                    + str(split[1])
                )
                args_to_pass[split[0]] = split[1]

        train(device, vit_custom(), constants, mapper, **args_to_pass)

    elif type_of_operation == "eval":
        print("Evaluate model")

        path_to_model_file = "" if len(sys.argv) < 3 else sys.argv[2]

        if path_to_model_file == "":
            raise Exception("No Model path specified")

        evaluate(device, path_to_model_file, mapper)

    elif type_of_operation == "show":
        path_to_tensor_file = "" if len(sys.argv) < 3 else sys.argv[2]

        if path_to_tensor_file == "":
            raise Exception("No Tensor path specified")

        show_image_from_transformed_stored_tensor(path_to_tensor_file)
    else:
        print("Unknown operation, please specify")

# Example model paths for my system

# path_to_model_file = "/media/jonas/69B577D0C4C25263/MLData/tensorboard/ML-Perceptron-RandSize/model_38.pth"
# path_to_model_file = "/media/jonas/69B577D0C4C25263/MLData/tensorboard/DINO-CLASSIFIER/momentum_0.9/model_29.pth"
# path_to_model_file = "/media/jonas/69B577D0C4C25263/MLData/tensorboard/DINO-CLASSIFIER/momentum_0/model_51.pth"
# path_to_model_file = "/media/jonas/69B577D0C4C25263/MLData/tensorboard/DINO-TINY/momentum_0/model_60.pth"
# path_to_model_file = "/media/jonas/69B577D0C4C25263/MLData/tensorboard/DINO-TINY/momentum_0.1/model_54.pth"
# path_to_model_file = "/media/jonas/69B577D0C4C25263/MLData/tensorboard/DINO-TINY/momentum_0.9/model_24.pth"

# Example show paths for my system

# path_to_tensor_file = "/media/jonas/69B577D0C4C25263/MLData/transformed2/val/n01443537_00032258.pt
