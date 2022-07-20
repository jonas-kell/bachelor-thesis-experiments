from math import inf
import numpy as np
from typing import Literal
import torch


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


def transform_torch_zero_matrix_to_neg_infinity(matrix: torch.Tensor) -> torch.Tensor:
    return torch.where(
        matrix == 0.0, torch.scalar_tensor(-inf, device=matrix.device), matrix
    )


def transform_adjacency_matrix(
    matrix: np.ndarray,
    neg_inf: bool = True,
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


def adjacency_matrix_from_locally_applied_kernel(
    kernel: np.ndarray, rows: int, columns: int
) -> np.ndarray:
    assert rows > 0
    assert columns > 0
    assert kernel.dtype == np.bool8
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 == 1

    result = np.zeros((rows * columns, rows * columns), dtype=np.bool8)
    k = (kernel.shape[0] - 1) // 2

    for i in range(rows):  # row index
        for j in range(columns):  # column index
            for off_i in range(-k, k + 1):  # row index
                for off_j in range(-k, k + 1):  # column indexs
                    # avoid shooting over boundaries
                    if (0 <= (i + off_i) < rows) and (0 <= (j + off_j) < columns):
                        # arranged like this (and so on)
                        #
                        # [[0,1,2],
                        #  [3,4,5]]

                        result[
                            i * columns + j,
                            (i + off_i) * columns + (j + off_j),
                        ] = kernel[off_i + k, off_j + k]

    return result


def self_matrix(n: int):
    assert n > 0

    return np.eye(n * n, dtype=np.bool8)


def nn_matrix(n: int):
    assert n > 0
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.bool8)

    return adjacency_matrix_from_locally_applied_kernel(kernel, n, n)


def nnn_matrix(n: int):
    assert n > 0
    kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.bool8)

    return adjacency_matrix_from_locally_applied_kernel(kernel, n, n)
