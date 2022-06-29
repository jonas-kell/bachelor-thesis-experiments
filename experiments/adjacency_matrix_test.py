import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
helper_dir = os.path.join(script_dir, "../models/helpers")
sys.path.append(helper_dir)

from adjacency_matrix import (
    nn_matrix,
    nnn_matrix,
    transform_adjacency_matrix,
    expand_by_one_unit,
)

adjacency_matrix = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.bool8)
adjacency_matrix = nn_matrix(224 // 16)

print(transform_adjacency_matrix(adjacency_matrix, True, "avg+1"))

nn = transform_adjacency_matrix(nn_matrix(3), False, "avg+1")
nnn = transform_adjacency_matrix(nnn_matrix(3), False, "avg+1")

# "proof" by validation, this can be pre-computed
# (it can be for sure, because of the addition and multiplication of matrices is distributive)
test = np.array(range(9))
print(0.5 * nn @ test + 0.2 * nnn @ test)
print((0.5 * nn + 0.2 * nnn) @ test)


test_matrix = np.array([[2, 31, 2], [2, -31, 2], [12, 1, 12]], dtype=np.float16)
print(test_matrix)
print(expand_by_one_unit(test_matrix))
