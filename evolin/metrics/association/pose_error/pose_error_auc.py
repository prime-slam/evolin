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

from typing import Tuple

from evolin.typing import ArrayN

__all__ = ["angular_pose_error_auc"]


def cumulative_error_curve(
    errors: ArrayN[float],
) -> Tuple[ArrayN[float], ArrayN[float]]:
    x = np.sort(errors)
    y = (np.arange(len(errors)) + 1) / len(errors)
    x = np.insert(x, 0, 0.0)
    y = np.insert(y, 0, 0.0)
    return x, y


def angular_pose_error_auc(
    rotation_errors: ArrayN[float], translation_errors: ArrayN[float], threshold: float
) -> float:
    """
    Calculates Area Under Cumulative Pose Error Curve.

    Parameters
    ----------
    rotation_errors
        array of angular rotation errors
    translation_errors
        array of angular translation errors
    threshold
        value on the x-axis up to which the area under the curve is calculated

    Returns
    -------
    value
        Area Under Cumulative Pose Error Curve to x-axis threshold

    Notes
    -----
    To construct a cumulative error curve,
    errors of the form min(rotation_errors[i], translation_errors[i]) are used.

    Examples
    --------
    >>> import numpy as np
    >>> rotation_errors = np.array([0, 0.15, 0.22, 0.89, 0.37, 0.44, 0.53])
    >>> translation_errors = np.array([0, 0.12, 0.25, 0.81, 0.30, 0.49, 0.5])
    >>> threshold = 0.4
    >>> pose_error_auc = angular_pose_error_auc(rotation_errors, translation_errors, threshold)
    """
    errors = np.min(np.column_stack([rotation_errors, translation_errors]), axis=-1)
    x, y = cumulative_error_curve(errors)
    last_index = np.searchsorted(x, threshold)
    return (
        np.trapz(
            x=np.append(x[:last_index], threshold),
            y=np.append(y[:last_index], y[last_index - 1]),
        )
        / threshold
    )
