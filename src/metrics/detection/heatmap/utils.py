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

from scipy.sparse import dok_matrix
from skimage.draw import line_nd
from typing import List, Sized

from src.typing import ArrayNx4


def equally_sized(batches: List[Sized]) -> bool:
    expected_size = len(batches[0])
    return all(expected_size == len(batch) for batch in batches)


def rasterize(lines: ArrayNx4[float], height: int, width: int) -> dok_matrix:
    bitmap = dok_matrix((height, width), dtype=bool)
    x_index = [0, 2]
    y_index = [1, 3]

    if lines.size != 0:
        lines = lines.astype(int)
        lines[:, x_index] = np.clip(lines[:, x_index], 0, width - 1)
        lines[:, y_index] = np.clip(lines[:, y_index], 0, height - 1)

        for x1, y1, x2, y2 in lines:
            end_point, start_point = line_nd((x1, y1), (x2, y2), endpoint=True)
            bitmap[start_point, end_point] = True

    return bitmap
