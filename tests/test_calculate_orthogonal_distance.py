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

from evolin.metrics.detection.vectorized.distance.orthogonal import OrthogonalDistance


def test_infinite_distance():
    first_lines = np.array([[[0, 0], [1, 0]]])
    second_lines = np.array([[[0, 0], [0, 1]]])
    expected_distance = np.array([[np.inf]])
    actual_distance = OrthogonalDistance().calculate(first_lines, second_lines)
    assert (actual_distance == expected_distance).all()


@pytest.mark.parametrize(
    "first_lines, second_lines, expected_distance",
    [
        (np.array([[[0, 0], [1, 0]]]), np.array([[[0, 5], [1, 5]]]), np.array([[10]])),
        (
            np.array([[[1, 0], [0, 1]]]),
            np.array([[[1, 0], [1, 1]], [[0, 1], [1, 0]]]),
            np.array([[0.8535, 0]]),
        ),
        (
            np.array([[[3, 0], [3, 4]]]),
            np.array([[[3, 4], [0, 0]], [[3, 0], [3, 4]]]),
            np.array([[2.7, 0.0]]),
        ),
    ],
)
def test_correct_finite_distance(first_lines, second_lines, expected_distance):
    actual_distance = OrthogonalDistance().calculate(first_lines, second_lines)
    diff = np.abs(actual_distance - expected_distance)
    eps = 0.001
    assert (diff < eps).all()
