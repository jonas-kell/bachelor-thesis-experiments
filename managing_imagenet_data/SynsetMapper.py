from PathAndFolderConstants import PathAndFolderConstants


class SynsetMapper:
    def __init__(self, constants: PathAndFolderConstants):
        # allocate categories
        f = open(constants.path_to_synset_mapping_file, "r")
        contents = f.read()

        lines = contents.split("\n", constants.nr_categories)

        self.synset_ids = [0] * constants.nr_categories
        self.synset_descriptions = [0] * constants.nr_categories
        for i in range(constants.nr_categories):
            category = lines[i].split(" ", 1)
            self.synset_ids[i] = category[0]
            self.synset_descriptions[i] = category[1]

    # get the numerical/vector index to a synset id
    def vector_index_from_synset_id(self, synset_id):
        return self.synset_ids.index(synset_id)

    # get the description to a synset id
    def description_from_synset_id(self, synset_id):
        return self.synset_descriptions[self.vector_index_from_synset_id(synset_id)]

    # get the synset id to a numerical/vector index
    def synset_id_from_vector_index(self, vector_index):
        return self.synset_ids[vector_index]

    # get the description to a numerical/vector index
    def description_from_vector_index(self, vector_index):
        return self.synset_descriptions[vector_index]

    def all_used_synset_ids(self):
        return self.synset_ids
