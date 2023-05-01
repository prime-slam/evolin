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

from evolin.typing import Array3, Array3x3, Array4x4


def make_euclidean_transform(
    rotation_matrix: Array3x3[float], translation: Array3[float]
) -> Array4x4[float]:
    transform = np.hstack([rotation_matrix, translation.reshape(-1, 1)])
    transform = np.vstack([transform, [0, 0, 0, 1]])
    return transform


def make_homogeneous_matrix(matrix: Array3x3[float]) -> Array4x4[float]:
    matrix_homo = np.hstack([matrix, np.zeros((3, 1))])
    matrix_homo = np.vstack([matrix_homo, [0, 0, 0, 1]])
    return matrix_homo
