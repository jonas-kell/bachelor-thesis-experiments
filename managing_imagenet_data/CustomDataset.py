import torch
import os
from custom_imagenet_constants import (
    train_folder_name,
    val_folder_name,
)
from mapping import vector_index_from_synset_id


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, mode="train"):
        if mode not in [train_folder_name, val_folder_name]:
            raise Exception(
                "only modes '"
                + train_folder_name
                + "' and '"
                + val_folder_name
                + "' are supported"
            )
        self.mode = mode

        if not mode in os.listdir(root_folder_path):
            raise Exception("folder '" + mode + "' not present in the root folder")
        self.root_folder_path = root_folder_path

        self.folder_path = os.path.join(self.root_folder_path, self.mode)

        self.files = os.listdir(self.folder_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(
            os.path.join(self.folder_path, self.files[idx])
        )  # load the features of this sample

        label = self.files[idx].split("_", 1)[0]
        class_id = vector_index_from_synset_id(label)

        return (sample, class_id)
