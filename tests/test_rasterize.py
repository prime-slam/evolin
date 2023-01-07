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
import pytest

from src.metrics.detection.heatmap.utils import rasterize


@pytest.mark.parametrize(
    "lines, height, width, expected_bitmap",
    [
        (
            np.array([[5, 0, 0, 5], [0, 0, 5, 5]]),
            5,
            5,
            np.diag(np.full(5, True)) + np.flip(np.diag(np.full(5, True)), axis=1),
        ),
        (
            np.array([[0, 0, 15, 0]]),
            3,
            5,
            np.array([[True] * 5, [False] * 5, [False] * 5]),
        ),
        (
            np.array([]),
            5,
            5,
            np.zeros((5, 5), bool),
        ),
    ],
)
def test_rasterize(lines, height, width, expected_bitmap):
    actual_bitmap = rasterize(lines, height, width)
    assert (actual_bitmap == expected_bitmap).all()
