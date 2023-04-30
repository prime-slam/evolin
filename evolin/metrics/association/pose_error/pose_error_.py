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

from evolin.typing import Array3x3, Array3, Array4x4
from typing import Tuple

__all__ = [
    "pose_error",
    "absolute_translation_error",
    "angular_translation_error",
    "angular_rotation_error",
]


def angular_rotation_error(
    first_rotation: Array3x3[float], second_rotation: Array3x3[float]
) -> float:
    """
    Calculates angular rotation error between two rotations.

    Parameters
    ----------
    first_rotation
        first rotation matrix
    second_rotation
        second rotation matrix

    Returns
    -------
    value
        angle between two rotations

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.transform import Rotation
    >>> axis = np.array([0, 0, 1])
    >>> first_angle = np.pi / 2
    >>> second_angle = np.pi / 3
    >>> first_rotation = Rotation.from_rotvec(first_angle * axis).as_matrix()
    >>> second_rotation = Rotation.from_rotvec(second_angle * axis).as_matrix()
    >>> error = angular_rotation_error(first_rotation, second_rotation)
    """
    cos = (np.trace(first_rotation @ second_rotation.T) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    return np.rad2deg(np.abs(np.arccos(cos)))


def angular_translation_error(
    first_translation: Array3[float], second_translation: Array3[float]
) -> float:
    """
    Calculates angular translation error.

    Parameters
    ----------
    first_translation
        first translation vector
    second_translation
        second translation vector

    Returns
    -------
    value
        angle between two translation vectors

    Examples
    --------
    >>> import numpy as np
    >>> first_translation = np.array([0, 1, 0])
    >>> second_translation = np.array([0, 0, 1])
    >>> error = angular_translation_error(first_translation, second_translation)
    """
    n = np.linalg.norm(first_translation) * np.linalg.norm(second_translation)
    return np.rad2deg(
        np.arccos(np.clip(np.dot(first_translation, second_translation) / n, -1.0, 1.0))
    )


def absolute_translation_error(
    pose_gt: Array4x4[float], pose_est: Array4x4[float]
) -> float:
    """
    Calculates absolute translation error between ground truth and extimated poses.

    Parameters
    ----------
    pose_gt
        ground truth transformation matrix
    pose_est
        estimated transformation matrix

    Returns
    -------
    value
        translation vector norm after applying reverse pose_est transformation

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.transform import Rotation
    >>> translation_gt = np.array([0, 1, 0])
    >>> translation_est = np.array([0, 0, 1])
    >>> axis = np.array([0, 0, 1])
    >>> angle_gt = np.pi / 2
    >>> angle_est = np.pi / 3
    >>> rotation_gt = Rotation.from_rotvec(angle_gt * axis).as_matrix()
    >>> rotation_est = Rotation.from_rotvec(angle_est * axis).as_matrix()
    >>> # euclidean transformation creation
    >>> pose_gt = np.vstack([
    >>>     np.hstack([rotation_gt, translation_gt.reshape(-1, 1)]),
    >>>     [0, 0, 0, 1]
    >>> ])
    >>> pose_est = np.vstack([
    >>>     np.hstack([rotation_est, translation_est.reshape(-1, 1)]),
    >>>     [0, 0, 0, 1]
    >>> ])
    >>> error = absolute_translation_error(pose_gt, pose_est)
    """
    delta_pose = pose_gt @ np.linalg.inv(pose_est)
    delta_translation = delta_pose[:3, 3]
    return np.linalg.norm(delta_translation)


def pose_error(
    pose_gt: Array4x4[float], pose_est: Array4x4[float]
) -> Tuple[float, float, float]:
    """
    Calculates pose errors.

    Parameters
    ----------
    pose_gt
        ground truth transformation matrix
    pose_est
        estimated transformation matrix

    Returns
    -------
    values
        angular_translation_error, angular_rotation_error, and absolute_translation_error

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.transform import Rotation
    >>> translation_gt = np.array([0, 1, 0])
    >>> translation_est = np.array([0, 0, 1])
    >>> axis = np.array([0, 0, 1])
    >>> angle_gt = np.pi / 2
    >>> angle_est = np.pi / 3
    >>> rotation_gt = Rotation.from_rotvec(angle_gt * axis).as_matrix()
    >>> rotation_est = Rotation.from_rotvec(angle_est * axis).as_matrix()
    >>> # euclidean transformation creation
    >>> pose_gt = np.vstack([
    >>>     np.hstack([rotation_gt, translation_gt.reshape(-1, 1)]),
    >>>     [0, 0, 0, 1]
    >>> ])
    >>> pose_est = np.vstack([
    >>>     np.hstack([rotation_est, translation_gt.reshape(-1, 1)]),
    >>>     [0, 0, 0, 1]
    >>> ])
    >>> (
    >>>     angular_translation_error_,
    >>>     angular_rotation_error_,
    >>>     absolute_translation_error_
    >>> ) = pose_error(pose_gt, pose_est)
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
