class PathAndFolderConstants:
    def __init__(
        self,
        path_to_imagenet_data_folder: str,
        path_to_folder_for_transformed_data: str,
        path_to_validation_solution_file: str,
        path_to_tensorboard_log_folder: str,
        nr_categories: int = 100,
        path_to_synset_mapping_file: str = "managing_imagenet_data/used_synsets.txt",
        train_folder_name: str = "train",
        val_folder_name: str = "val",
    ):
        self.path_to_imagenet_data_folder = path_to_imagenet_data_folder
        self.path_to_folder_for_transformed_data = path_to_folder_for_transformed_data
        self.path_to_validation_solution_file = path_to_validation_solution_file
        self.path_to_tensorboard_log_folder = path_to_tensorboard_log_folder
        self.nr_categories = nr_categories
        self.path_to_synset_mapping_file = path_to_synset_mapping_file
        self.train_folder_name = train_folder_name
        self.val_folder_name = val_folder_name
