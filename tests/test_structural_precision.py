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

from src.metrics.detection.vectorized import vectorized_precision


@pytest.mark.parametrize(
    "pred_lines, gt_lines, expected",
    [
        (
            [np.array([[1, 0, 2, 0], [2, 0, 1, 0], [5, 9, 11, 13]])],
            [np.array([[1, 0, 2, 0], [5, 10, 11, 12]])],
            0.6666,
        ),
        ([np.array([[1, 0, 2, 0]])], [np.array([[1, 2, 2, 2]])], 0),
        (
            [
                np.array([[1, 0, 2, 0], [2, 0, 1, 0], [5, 9, 11, 13]]),
                np.array([[1, 0, 2, 0]]),
            ],
            [np.array([[1, 0, 2, 0], [5, 10, 11, 12]]), np.array([[1, 2, 2, 2]])],
            0.5,
        ),
    ],
)
def test_structural_precision(pred_lines, gt_lines, expected):
    actual = vectorized_precision(pred_lines, gt_lines, distance="structural")
    eps = 0.001
    assert np.abs(actual - expected) < eps
