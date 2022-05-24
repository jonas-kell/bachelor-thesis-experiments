import os
from custom_imagenet_constants import (
    path_to_imagenet_data,
    path_to_target_folder_for_transformed_data,
    train_folder_name,
    val_folder_name,
    nr_categories,
    path_to_validation_solution_file,
)
from mapping import all_used_synset_ids, synset_id_from_vector_index
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
from alive_progress import alive_bar


def transform_training_data(transformation):
    if not train_folder_name in os.listdir(path_to_imagenet_data):
        raise Exception(
            "the folder must directly contain the 'train' folder in the format of imagenet"
        )

    path_to_train_target_folder = os.path.join(
        path_to_target_folder_for_transformed_data, train_folder_name
    )
    if not train_folder_name in os.listdir(path_to_target_folder_for_transformed_data):
        os.mkdir(path_to_train_target_folder)
    if len(os.listdir(path_to_train_target_folder)) != 0:
        raise Exception("The 'train' folder in the target directory must be empty")

    # total number of images
    total_nr_images = 0
    for idx in range(nr_categories):
        synset_id = synset_id_from_vector_index(idx)

        folder_with_training_data = os.path.join(
            path_to_imagenet_data, train_folder_name, synset_id
        )

        image_file_names = os.listdir(folder_with_training_data)

        total_nr_images += len(image_file_names)

    # transformations
    with alive_bar(total_nr_images) as bar:
        for idx in range(nr_categories):
            synset_id = synset_id_from_vector_index(idx)

            folder_with_training_data = os.path.join(
                path_to_imagenet_data, train_folder_name, synset_id
            )

            image_file_names = os.listdir(folder_with_training_data)

            for image_file_name in image_file_names:
                name = image_file_name.split(".", 1)[0]
                path_to_file = os.path.join(folder_with_training_data, image_file_name)

                img = Image.open(path_to_file)

                if img.mode == "RGB":  # grayscale images are ignored for simplicity

                    tensor = transformation(img)
                    torch.save(
                        tensor,
                        os.path.join(
                            path_to_target_folder_for_transformed_data,
                            train_folder_name,
                            name + ".pt",
                        ),
                    )

                img.close()  # free image memory

                bar()  # advance progress bar


def transform_validation_data(transformation):
    if not val_folder_name in os.listdir(path_to_imagenet_data):
        raise Exception(
            "the folder must directly contain the 'val' folder in the format of imagenet"
        )

    path_to_validation_target_folder = os.path.join(
        path_to_target_folder_for_transformed_data, val_folder_name
    )
    if not val_folder_name in os.listdir(path_to_target_folder_for_transformed_data):
        os.mkdir(path_to_validation_target_folder)
    if len(os.listdir(path_to_validation_target_folder)) != 0:
        raise Exception("The 'val' folder in the target directory must be empty")

    # total number of images
    total_nr_images = len(
        os.listdir(os.path.join(path_to_imagenet_data, val_folder_name))
    )

    # pre-allocate info about the solutions
    f = open(path_to_validation_solution_file, "r")
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
            path_to_imagenet_data, val_folder_name
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
                if solution_synset in all_used_synset_ids():
                    tensor = transformation(img)
                    torch.save(
                        tensor,
                        os.path.join(
                            path_to_target_folder_for_transformed_data,
                            val_folder_name,
                            solution_synset + "_" + val_index + ".pt",
                        ),
                    )

            img.close()  # free image memory

            bar()  # advance progress bar


# show a Tensor as image
def show_image_from_transformed_stored_tensor(path):
    tensor = torch.load(path)

    inv_trans = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )

    plt.imshow(transforms.ToPILImage()(inv_trans(tensor)))
    plt.waitforbuttonpress()


if __name__ == "__main__":
    transformation = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            # these are the image net means ->
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ],
    )

    # transform_training_data(transformation)
    # transform_validation_data(transformation)

    # show_image_from_transformed_stored_tensor(
    #     "/media/jonas/69B577D0C4C25263/MLData/transformed/val/n01806143_00037524.pt"
    # )
