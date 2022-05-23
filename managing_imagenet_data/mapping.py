# define lookup file
synset_mapping_name = "used_synsets.txt"
nr_categories = 100


# allocate categories
f = open(synset_mapping_name, "r")
contents = f.read()

lines = contents.split("\n", nr_categories)

synset_ids = [0] * nr_categories
synset_descriptions = [0] * nr_categories
for i in range(nr_categories):
    category = lines[i].split(" ", 1)
    synset_ids[i] = category[0]
    synset_descriptions[i] = category[1]

# get the numerical/vector index to a synset id
def vector_index_from_synset_id(synset_id):
    return synset_ids.index(synset_id)


# get the description to a synset id
def description_from_synset_id(synset_id):
    return synset_descriptions[vector_index_from_synset_id(synset_id)]


# get the synset id to a numerical/vector index
def vector_index_from_synset_id(vector_index):
    return synset_ids[vector_index]


# get the description to a numerical/vector index
def description_from_vector_index(vector_index):
    return synset_descriptions[vector_index]
