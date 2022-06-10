import torch
import os

from PathAndFolderConstants import PathAndFolderConstants
from SynsetMapper import SynsetMapper


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self, constants: PathAndFolderConstants, mapper: SynsetMapper, mode="train"
    ):
        self.mapper = mapper

        if mode not in [
            constants.train_folder_name,
            constants.val_folder_name,
        ]:
            raise Exception(
                "only modes '"
                + constants.train_folder_name
                + "' and '"
                + constants.val_folder_name
                + "' are supported"
            )
        self.mode = mode

        if not mode in os.listdir(constants.path_to_folder_for_transformed_data):
            raise Exception(
                "folder '" + mode + "' not present in the preprocessed-data folder"
            )
        self.root_folder_path = constants.path_to_folder_for_transformed_data

        self.folder_path = os.path.join(self.root_folder_path, self.mode)

        self.files = os.listdir(self.folder_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(
            os.path.join(self.folder_path, self.files[idx])
        )  # load the features of this sample

        label = self.files[idx].split("_", 1)[0]
        class_id = self.mapper.vector_index_from_synset_id(label)

        return (sample, class_id)
