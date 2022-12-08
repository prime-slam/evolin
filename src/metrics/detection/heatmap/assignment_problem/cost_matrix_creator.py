# Copyright (c) 2022, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from src.typing import ArrayNx3, ArrayNxN


class CostMatrixCreator:
    def __init__(self, outlier_cost, outlier_degree=6):
        self.outlier_cost = outlier_cost
        self.outlier_degree = outlier_degree

    def create(
        self, edges: ArrayNx3[np.float], first_part_size: int, second_part_size: int
    ) -> ArrayNxN[np.float]:
        matrix_size = first_part_size + second_part_size
        min_part_size = min(first_part_size, second_part_size)
        max_part_size = max(first_part_size, second_part_size)
        input_matrix = np.zeros((matrix_size, matrix_size))

        first_part_outlier_degree = max(
            0, min(self.outlier_degree, first_part_size - 1)
        )
        second_part_outlier_degree = max(
            0, min(self.outlier_degree, second_part_size - 1)
        )
        outlier_to_outlier_degree = min(
            self.outlier_degree, first_part_size, second_part_size
        )

        # add real edges
        start_nodes = edges[:, 0].astype(int)
        end_nodes = edges[:, 1].astype(int)
        weights = edges[:, 2]

        input_matrix[start_nodes, end_nodes] = weights

        # add outlier edges for first part excluding diagonal
        for i in np.arange(first_part_size):
            j = np.random.choice(
                first_part_size - 1, first_part_outlier_degree, replace=False
            )
            j[j >= i] += 1
            input_matrix[i, second_part_size + j] = self.outlier_cost

        # add outlier edges for second part excluding diagonal
        for j in np.arange(second_part_size):
            i = np.random.choice(
                second_part_size - 1, second_part_outlier_degree, replace=False
            )
            i[i >= j] += 1
            input_matrix[first_part_size + i, j] = self.outlier_cost

        # add outlier-to-outlier edges
        for i in np.arange(max_part_size):
            j = np.random.choice(
                min_part_size, outlier_to_outlier_degree, replace=False
            )
            if first_part_size < second_part_size:
                input_matrix[
                    first_part_size + i, second_part_size + j
                ] = self.outlier_cost
            else:
                input_matrix[
                    first_part_size + j, second_part_size + i
                ] = self.outlier_cost

        # add perfect match overlay
        lower_overlay_start_nodes = np.arange(first_part_size)
        lower_overlay_end_nodes = np.arange(first_part_size) + second_part_size
        upper_overlay_start_nodes = first_part_size + np.arange(second_part_size)
        upper_overlay_end_nodes = np.arange(second_part_size)

        input_matrix[
            lower_overlay_start_nodes, lower_overlay_end_nodes
        ] = self.outlier_cost
        input_matrix[
            upper_overlay_start_nodes, upper_overlay_end_nodes
        ] = self.outlier_cost

        return input_matrix
