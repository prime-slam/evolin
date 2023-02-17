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

from src.typing import Array3x3, Array3, Array4x4

__all__ = [
    "pose_error",
    "absolute_translation_error",
    "angular_translation_error",
    "angular_rotation_error",
]


def angular_rotation_error(
    first_rotation: Array3x3[float], second_rotation: Array3x3[float]
):
    """
    Calculates angular rotation error
    :param first_rotation: first rotation matrix
    :param second_rotation: second rotation matrix
    :return: angle between two rotations
    """
    cos = (np.trace(first_rotation @ second_rotation.T) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    return np.rad2deg(np.abs(np.arccos(cos)))


def angular_translation_error(
    first_translation: Array3[float], second_translation: Array3[float]
):
    """
    Calculates angular translation error
    :param first_translation: first translation vector
    :param second_translation: second translation vector
    :return: angle between two translation vectors
    """
    n = np.linalg.norm(first_translation) * np.linalg.norm(second_translation)
    return np.rad2deg(
        np.arccos(np.clip(np.dot(first_translation, second_translation) / n, -1.0, 1.0))
    )


def absolute_translation_error(pose_gt: Array4x4[float], pose_est: Array4x4[float]):
    """
    Calculates absolute translation error
    :param pose_gt: ground truth transformation matrix
    :param pose_est: estimated transformation matrix
    :return: translation vector norm after applying reverse pose_est transformation
    and then pose_gt
    """
    delta_pose = pose_gt @ np.linalg.inv(pose_est)
    delta_translation = delta_pose[:3, 3]
    return np.linalg.norm(delta_translation)


def pose_error(pose_gt, pose_est):
    """
    Calculates pose errors
    :param pose_gt: ground truth transformation matrix
    :param pose_est: estimated transformation matrix
    :return: angular_translation_error, angular_rotation_error
    and absolute_translation_error
    """
    rotation_est = pose_est[:3, :3]
    translation_est = pose_est[:3, 3]
    rotation_gt = pose_gt[:3, :3]
    translation_gt = pose_gt[:3, 3]
    angular_translation_error_ = angular_translation_error(
        translation_est, translation_gt
    )
    angular_rotation_error_ = angular_rotation_error(rotation_est, rotation_gt)
    absolute_translation_error_ = absolute_translation_error(pose_gt, pose_est)

    return (
        angular_translation_error_,
        angular_rotation_error_,
        absolute_translation_error_,
    )
