import inspect
import sys
import os
from typing import Literal
import torch
from torch import nn
import torch
import random
import numpy as np
import re
from alive_progress import alive_bar

sys.path.append(os.path.abspath("./managing_imagenet_data"))
sys.path.append(os.path.abspath("./slack"))
sys.path.append(os.path.abspath("./train"))
sys.path.append(os.path.abspath("./models"))

from train.training_script import train_model
from managing_imagenet_data.pre_process_data import (
    transform_validation_data,
    transform_training_data,
    show_image_from_transformed_stored_tensor,
    show_image_from_transformed_stored_image,
)
from transformations import resize, resize_normalize_to_tensor
from PathAndFolderConstants import PathAndFolderConstants
from SynsetMapper import SynsetMapper
from evaluating.evaluate_model import evaluate_model
from train.checkpoint import get_highest_epoch, get_hyperparameter_dict

constants = PathAndFolderConstants(
    path_to_imagenet_data_folder="/media/jonas/69B577D0C4C25263/MLData/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC",
    path_to_folder_for_transformed_data="/media/jonas/69B577D0C4C25263/MLData/transformed",
    path_to_validation_solution_file="/media/jonas/69B577D0C4C25263/MLData/imagenet-object-localization-challenge/LOC_val_solution.csv",
    path_to_tensorboard_log_folder="/media/jonas/69B577D0C4C25263/MLData/tensorboard_trash",
)
mapper = SynsetMapper(constants)

from models.mlp import NeuralNetwork
from models.vision_transformer import vit_custom, vit_tiny
from models.res_net import res_net, res_net_pretrained
from models.sail_sg_poolformer import poolformer_s12
from models.metaformer import (
    vision_transformer,
    graph_vision_transformer_nn,
    graph_vision_transformer_nnn,
    poolformer,
    graph_poolformer_nn,
    graph_poolformer_nnn,
    depthwise_conformer,
    symmetric_depthwise_conformer_nn,
    symmetric_depthwise_conformer_nnn,
    symmetric_graph_depthwise_conformer_nn,
    symmetric_graph_depthwise_conformer_nnn,
    full_conformer_nnn,
)

available_models = {  # add custom configurations in this dict
    "ML-Perceptron-RandSize": NeuralNetwork,
    "RES-NET": res_net,
    "RES-NET-PRETRAINED": res_net_pretrained,
    "PAPER-POOLFORMER": poolformer_s12,
    "DINO-TINY": vit_tiny,
    "DINO-CLASSIFIER": vit_custom,
    "VT": vision_transformer,
    "GVT-NN": graph_vision_transformer_nn,
    "GVT-NNN": graph_vision_transformer_nnn,
    "PF": poolformer,
    "GP-NN": graph_poolformer_nn,
    "GP-NNN": graph_poolformer_nnn,
    "CD": depthwise_conformer,
    "SD-NN": symmetric_depthwise_conformer_nn,
    "SD-NNN": symmetric_depthwise_conformer_nnn,
    "GSD-NN": symmetric_graph_depthwise_conformer_nn,
    "GSD-NNN": symmetric_graph_depthwise_conformer_nnn,
    "CF": full_conformer_nnn,
}


def prepare_data(
    constants: PathAndFolderConstants,
    mapper: SynsetMapper,
):
    transformation = resize

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


def cleanup_models(
    constants: PathAndFolderConstants,
):
    predicate = re.compile("model_(\d+)\.pth")
    predicate_keep = re.compile("keep_(\d+)")

    tensorboard_folder = constants.path_to_tensorboard_log_folder
    log_folders = os.listdir(tensorboard_folder)

    for folder in log_folders:
        full_path = os.path.join(tensorboard_folder, folder)

        print(f"Cleaning the folder '{full_path}'")

        folder_contents = os.listdir(full_path)

        filtered = [
            int(re.search(r"model_(\d+)\.pth", s).group(1))
            for s in folder_contents
            if predicate.match(s)
        ]

        keep = [
            int(re.search(r"keep_(\d+)", s).group(1))
            for s in folder_contents
            if predicate_keep.match(s)
        ]

        print(f"Found {len(filtered)} 'model_##.pth' files")
        max_index = max(filtered, default=-1)

        print(f"Manually set to keep indices {keep}")

        to_throw_away = [
            f"model_{index}.pth"
            for index in filtered
            if index % 10 != 9 and index != max_index and index not in keep
        ]

        print(f"Deleting {len(to_throw_away)} elements")

        with alive_bar(len(to_throw_away)) as bar:

            for throw_away_model in to_throw_away:
                path = os.path.join(full_path, throw_away_model)

                os.remove(path)

                bar()  # advance


def evaluate(
    device: Literal["cuda", "cpu"],
    path_to_model_file: str,
    mapper: SynsetMapper,
):

    print("Using model: " + path_to_model_file + " to evaluate images")

    hyperparam_dict = os.path.dirname(path_to_model_file)
    backloaded_params = get_hyperparameter_dict(hyperparam_dict)

    model_name = backloaded_params["model_name"]

    model = available_models[model_name]()

    evaluate_model(
        device, model, path_to_model_file, resize_normalize_to_tensor, mapper
    )


if __name__ == "__main__":
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    type_of_operation = "" if len(sys.argv) == 1 else sys.argv[1]

    if type_of_operation == "prep":
        print("Prepare the data")

        prepare_data(constants, mapper)

    if type_of_operation == "cleanup":
        print("Cleanup obsolete model files")

        cleanup_models(constants)

    elif type_of_operation == "train":
        # training requires random numbers.
        # They will be encountered in this file already, because here random numbers may be used in the creation of the neural networks
        # (the functions are in the dict)
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        # eval script
        print("Train model")

        additional_parameters = [] if len(sys.argv) < 3 else sys.argv[2:]

        supported_args = inspect.getfullargspec(train_model).args
        types = inspect.getfullargspec(train_model).annotations
        args_to_pass = {}

        args_to_pass["model_name"] = list(available_models.keys())[0]

        continue_training = False
        continue_training_path = ""

        for param_string in additional_parameters:
            split = param_string.split("=", 1)

            if len(split) == 2:
                if split[0] == "continue":
                    continue_training = True
                    continue_training_path = str(split[1])
                if split[0] in supported_args:
                    if types[str(split[0])] in [int, bool, float]:
                        val = types[str(split[0])](split[1])
                    else:
                        val = str(split[1])

                    print(
                        "Use additional parameter: "
                        + str(split[0])
                        + " with type "
                        + str(types[str(split[0])])
                        + " with value "
                        + str(val)
                    )
                    args_to_pass[split[0]] = val
                if split[0] == "model_name":
                    args_to_pass["model_name"] = split[1]

        start_epoch = 0
        if continue_training:
            # load back params from prev session
            start_epoch = get_highest_epoch(continue_training_path) + 1
            print(f"Epoch to continue on: {start_epoch}")

            backloaded_params = get_hyperparameter_dict(continue_training_path)
            for param in backloaded_params:
                if (
                    param in supported_args and not param in args_to_pass
                ) or param == "model_name":  # allowed parameter, that is not already set
                    if param == "model_name":
                        args_to_pass["model_name"] = backloaded_params[param]
                        print(f"Model overwrite to: {backloaded_params[param]}")
                    else:
                        if types[param] in [int, bool, float]:
                            val = types[param](backloaded_params[param])
                        else:
                            val = str(backloaded_params[param])
                        print(f"Backloading Param '{param}' with value: {val}")
                        args_to_pass[param] = val

        if args_to_pass["model_name"] not in available_models.keys():
            raise Exception(
                "Model not configured. Try adding it to the list of supported run configurations above."
            )
        use_model = available_models[args_to_pass["model_name"]]
        print("Using the model: " + args_to_pass["model_name"])

        train(
            device,
            use_model(),
            constants,
            mapper,
            start_epoch=start_epoch,
            is_continuing_training=continue_training,
            continue_training_path=continue_training_path,
            **args_to_pass,
        )

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

    elif type_of_operation == "show_image":
        path_to_image_file = "" if len(sys.argv) < 3 else sys.argv[2]

        if path_to_image_file == "":
            raise Exception("No Image path specified")

        show_image_from_transformed_stored_image(path_to_image_file)

    else:
        print("Unknown operation, please specify")
