import torch
import os

from PathAndFolderConstants import PathAndFolderConstants
from SynsetMapper import SynsetMapper
from PIL import Image
import torchvision.transforms as transforms
from transformations import normalize_to_tensor


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        constants: PathAndFolderConstants,
        mapper: SynsetMapper,
        mode="train",
        preload_to_ram=True,
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

        self.to_tensor = normalize_to_tensor
        self.augmentation = transforms.Compose([transforms.RandomHorizontalFlip()])

        self.preload_to_ram = preload_to_ram
        if preload_to_ram:
            print("Preloading Dataset")
            self.images = [None] * len(self.files)

            total = len(self.files)
            five_percent = total // 20

            for idx in range(total):
                image = Image.open(os.path.join(self.folder_path, self.files[idx]))
                image.load()  # load the features of this sample
                self.images[idx] = image
                if idx % five_percent == 0:
                    print(str(5 * int(idx / five_percent)) + "%")

            print("Finished Preloading Dataset")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.preload_to_ram:
            sample = self.augmentation(self.to_tensor(self.images[idx]))
        else:
            image = Image.open(os.path.join(self.folder_path, self.files[idx]))
            sample = self.augmentation(self.to_tensor(image))
            image.close()

        label = self.files[idx].split("_", 1)[0]
        class_id = self.mapper.vector_index_from_synset_id(label)

        return (sample, class_id)
