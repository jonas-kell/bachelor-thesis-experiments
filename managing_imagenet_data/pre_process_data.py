import os
from PathAndFolderConstants import PathAndFolderConstants
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
from alive_progress import alive_bar
from SynsetMapper import SynsetMapper
from transformations import normalize_inverse


def transform_training_data(
    transformation, constants: PathAndFolderConstants, mapper: SynsetMapper
):
    if not constants.train_folder_name in os.listdir(
        constants.path_to_imagenet_data_folder
    ):
        raise Exception(
            "the folder must directly contain the 'train' folder in the format of imagenet"
        )

    path_to_train_target_folder = os.path.join(
        constants.path_to_folder_for_transformed_data, constants.train_folder_name
    )
    if not constants.train_folder_name in os.listdir(
        constants.path_to_folder_for_transformed_data
    ):
        os.mkdir(path_to_train_target_folder)
    if len(os.listdir(path_to_train_target_folder)) != 0:
        raise Exception("The 'train' folder in the target directory must be empty")

    # total number of images
    total_nr_images = 0
    for idx in range(constants.nr_categories):
        synset_id = mapper.synset_id_from_vector_index(idx)

        folder_with_training_data = os.path.join(
            constants.path_to_imagenet_data_folder,
            constants.train_folder_name,
            synset_id,
        )

        image_file_names = os.listdir(folder_with_training_data)

        total_nr_images += len(image_file_names)

    # transformations
    with alive_bar(total_nr_images) as bar:
        for idx in range(constants.nr_categories):
            synset_id = mapper.synset_id_from_vector_index(idx)

            folder_with_training_data = os.path.join(
                constants.path_to_imagenet_data_folder,
                constants.train_folder_name,
                synset_id,
            )

            image_file_names = os.listdir(folder_with_training_data)

            for image_file_name in image_file_names:
                name = image_file_name.split(".", 1)[0]
                path_to_file = os.path.join(folder_with_training_data, image_file_name)

                img = Image.open(path_to_file)

                if img.mode == "RGB":  # grayscale images are ignored for simplicity

                    image = transformation(img)
                    image.save(
                        os.path.join(
                            constants.path_to_folder_for_transformed_data,
                            constants.train_folder_name,
                            name + ".JPEG",
                        ),
                        "JPEG",
                        quality=80,
                        optimize=True,
                        progressive=True,
                    )

                img.close()  # free image memory

                bar()  # advance progress bar


def transform_validation_data(
    transformation, constants: PathAndFolderConstants, mapper: SynsetMapper
):
    if not constants.val_folder_name in os.listdir(
        constants.path_to_imagenet_data_folder
    ):
        raise Exception(
            "the folder must directly contain the 'val' folder in the format of imagenet"
        )

    path_to_validation_target_folder = os.path.join(
        constants.path_to_folder_for_transformed_data, constants.val_folder_name
    )
    if not constants.val_folder_name in os.listdir(
        constants.path_to_folder_for_transformed_data
    ):
        os.mkdir(path_to_validation_target_folder)
    if len(os.listdir(path_to_validation_target_folder)) != 0:
        raise Exception("The 'val' folder in the target directory must be empty")

    # total number of images
    total_nr_images = len(
        os.listdir(
            os.path.join(
                constants.path_to_imagenet_data_folder, constants.val_folder_name
            )
        )
    )

    # pre-allocate info about the solutions
    f = open(constants.path_to_validation_solution_file, "r")
    contents = f.read().splitlines()
    contents.pop(0)  # remove first line (csv descriptors)
    f.close()

    solution_synset_mapping = {}
    for i in range(len(contents)):
        split_line = contents[i].split(",", 1)
        solution_synset_mapping[split_line[0]] = split_line[1].split(" ", 1)[0]

    # transformations
    with alive_bar(total_nr_images) as bar:
        folder_with_validation_data = os.path.join(
            constants.path_to_imagenet_data_folder, constants.val_folder_name
        )

        image_file_names = os.listdir(folder_with_validation_data)

        for image_file_name in image_file_names:
            name = image_file_name.split(".", 1)[0]
            path_to_file = os.path.join(folder_with_validation_data, image_file_name)

            img = Image.open(path_to_file)

            if img.mode == "RGB":  # grayscale images are ignored for simplicity

                solution_synset = solution_synset_mapping[name]
                val_index = name.split("_", 2)[2]

                # only validation data for used synsets needed
                if solution_synset in mapper.all_used_synset_ids():

                    image = transformation(img)
                    image.save(
                        os.path.join(
                            constants.path_to_folder_for_transformed_data,
                            constants.val_folder_name,
                            solution_synset + "_" + val_index + ".JPEG",
                        ),
                        "JPEG",
                        quality=80,
                        optimize=True,
                        progressive=True,
                    )

            img.close()  # free image memory

            bar()  # advance progress bar


# show a Tensor as image
def show_image_from_transformed_stored_tensor(path):
    tensor = torch.load(path)

    plt.imshow(transforms.ToPILImage()(normalize_inverse(tensor)))
    plt.waitforbuttonpress()


# show a image as image (duh)
def show_image_from_transformed_stored_image(path):
    image = Image.open(path)
    image.load()

    to_tensor = transforms.Compose(
        [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)]
    )

    tensor = to_tensor(image)

    plt.imshow(transforms.ToPILImage()(normalize_inverse(tensor)))
    plt.waitforbuttonpress()
