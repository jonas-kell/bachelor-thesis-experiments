# pip install -U tbparse
from tbparse import SummaryReader
import os
import re


def get_hyperparameter_dict(path):
    dict = {}

    reader = SummaryReader(path)
    hp = reader.hparams

    tags = hp["tag"].tolist()

    for tag in tags:
        dict[tag] = hp.at[hp.index[hp["tag"] == tag][0], "value"]

    return dict


def get_highest_epoch(path):
    predicate = re.compile("model_(\d+)\.pth")
    filtered = [
        int(re.search(r"model_(\d+)\.pth", s).group(1))
        for s in os.listdir(path)
        if predicate.match(s)
    ]

    if len(filtered) > 0:
        max_epoch = max(filtered)
    else:
        max_epoch = 0

    return max_epoch


def get_stored_model_path_for_epoch(path, epoch):
    path = os.path.join(path, "model_" + str(epoch) + ".pth")

    if not os.path.exists(path) or not os.path.isfile(path):
        raise RuntimeError(
            f"Path '{path}' cannot be loaded as a model checkpoint as it doesn't exist as a file"
        )

    return path


if __name__ == "__main__":
    print(
        get_hyperparameter_dict(
            "/media/jonas/69B577D0C4C25263/MLData/tensorboard/RES-NET/not-pretrained (copy)"
        )
    )

    epoch = get_highest_epoch(
        "/media/jonas/69B577D0C4C25263/MLData/tensorboard/RES-NET/not-pretrained (copy)"
    )

    print(
        get_stored_model_path_for_epoch(
            "/media/jonas/69B577D0C4C25263/MLData/tensorboard/RES-NET/not-pretrained (copy)",
            epoch,
        )
    )
