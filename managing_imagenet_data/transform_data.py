import os
from custom_imagenet_constants import (
    path_to_imagenet_data,
    path_to_target_folder_for_transformed_data,
    train_folder_name,
    val_folder_name,
    nr_categories,
)
from mapping import synset_id_from_vector_index
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
from alive_progress import alive_bar


def transform_training_data(transformation):
    if not train_folder_name in os.listdir(path_to_imagenet_data):
        raise Exception(
            "the folder must directly conatin the 'train' folder in the format of imagenet"
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

    with alive_bar(total_nr_images) as bar:

        # transformations
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

    transform_training_data(transformation)

# show a Tensor as image

# to_image = transforms.ToPILImage()
# plt.imshow(to_image(tensor))
# plt.waitforbuttonpress()
