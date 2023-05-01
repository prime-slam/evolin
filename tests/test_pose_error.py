# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
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
from scipy.spatial.transform import Rotation

from evaluation.common.geometry.transform import make_euclidean_transform
from evolin.metrics.association.pose_error import (
    angular_rotation_error,
    angular_translation_error,
    absolute_translation_error,
)


@pytest.mark.parametrize(
    "first_angle, second_angle",
    [
        (
            np.pi / 2,
            np.pi / 3,
        ),
        (
            np.pi / 2,
            0,
        ),
        (
            0,
            np.pi,
        ),
        (
            0,
            0,
        ),
    ],
)
def test_angular_rotation_error(
    first_angle,
    second_angle,
):
    epsilon = 1e-5
    expected_angle_difference = np.rad2deg(np.abs(first_angle - second_angle))
    axis = np.array([0, 0, 1])
    actual_angle_difference = angular_rotation_error(
        Rotation.from_rotvec(first_angle * axis).as_matrix(),
        Rotation.from_rotvec(second_angle * axis).as_matrix(),
    )
    assert np.abs(expected_angle_difference - actual_angle_difference) < epsilon


@pytest.mark.parametrize(
    "first_translation, second_translation, expected_angle",
    [
        (np.array([0, 1, 0]), np.array([0, 0, 1]), 90),
        (np.array([0, 1, 0]), np.array([0, 1, 0]), 0),
        (np.array([0, 1, 1]), np.array([0, 0, 1]), 45),
    ],
)
def test_angular_translation_error(
    first_translation, second_translation, expected_angle
):
    epsilon = 1e-5
    actual_angle = angular_translation_error(first_translation, second_translation)
    assert np.abs(expected_angle - actual_angle) < epsilon


@pytest.mark.parametrize(
    "first_translation, second_translation, first_angle, second_angle",
    [
        (
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.pi / 2,
            np.pi / 3,
        ),
        (
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
            np.pi / 2,
            0,
        ),
        (
            np.array([0, 1, 1]),
            np.array([0, 0, 1]),
            0,
            np.pi,
        ),
    ],
)
def test_absolute_translation_error(
    first_translation,
    second_translation,
    first_angle,
    second_angle,
):
    epsilon = 1e-5
    axis = np.array([0, 0, 1])
    first_rotation = Rotation.from_rotvec(first_angle * axis).as_matrix()
    second_rotation = Rotation.from_rotvec(second_angle * axis).as_matrix()
    expected_norm = np.linalg.norm(
        -first_rotation @ second_rotation.T @ second_translation + first_translation
    )
    actual_norm = absolute_translation_error(
        make_euclidean_transform(first_rotation, first_translation),
        make_euclidean_transform(second_rotation, second_translation),
    )
    assert np.abs(expected_norm - actual_norm) < epsilon
