from typing import Literal
from xmlrpc.client import boolean
import numpy as np


def get_adding_matrix(matrix: np.ndarray) -> np.ndarray:
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.dtype == np.bool8

    return matrix.astype(np.float16)


def get_averaging_matrix(matrix: np.ndarray, count_own_connection=True) -> np.ndarray:
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.dtype == np.bool8

    n = matrix.shape[0]

    # init the matrix that counts the connections
    connections_count = np.zeros_like(matrix, dtype=np.float16)

    # for each connecting node, increase the count
    for i in range(n):
        for j in range(n):
            if i != j:
                connections_count[i, i] += matrix[i, j]
            elif count_own_connection:
                # add one if counting own connections
                connections_count[i, i] += 1

    # invert the matrix to be able to get  D -> D^(-1/2)
    # D^(-1/2) this notion breaks my brain, but is applicable here because the matrices are strict positive, diagonal and square
    for i in range(n):  # np functions do not play nice with my zeros -> loop
        connections_count[i, i] = connections_count[i, i] ** (-1 / 2)

    return connections_count @ matrix.astype(np.float16) @ connections_count


def transform_zero_matrix_to_neg_infinity(matrix: np.ndarray) -> np.ndarray:
    assert matrix.dtype == np.float16

    return np.where(matrix == 0.0, -np.Infinity, matrix)


def transform_adjacency_matrix(
    matrix: np.ndarray,
    neg_inf: boolean = True,
    type: Literal["sum", "avg", "avg+1"] = "sum",
) -> np.ndarray:
    if type == "sum":
        result = get_adding_matrix(matrix)
    if type == "avg":
        result = get_averaging_matrix(matrix, count_own_connection=False)
    if type == "avg+1":
        result = get_averaging_matrix(matrix, count_own_connection=True)

    if neg_inf:
        return transform_zero_matrix_to_neg_infinity(result)
    else:
        return result


if __name__ == "__main__":
    adjacency_matrix = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.bool8)
    print(transform_adjacency_matrix(adjacency_matrix, True, "avg+1"))
