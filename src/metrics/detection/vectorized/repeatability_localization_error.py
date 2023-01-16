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

from typing import List, Tuple, Union

import itertools as it
import numpy as np

from src.metrics.detection.vectorized import DISTANCE_NAMES, EVALUATION_RESOLUTION
from src.metrics.detection.vectorized.distance.distance import Distance
from src.metrics.detection.vectorized.distance.distance_factory import (
    DistanceFactory,
)
from src.metrics.detection.vectorized.utils import (
    docstring_arg,
)
from src.typing import ArrayNx4

__all__ = ["repeatability", "localization_error", "repeatability_localization_error"]


@docstring_arg(DISTANCE_NAMES)
def repeatability_localization_error(
    first_lines_batch: List[ArrayNx4[float]],
    second_lines_batch: List[ArrayNx4[float]],
    first_lines_projections_batch: List[ArrayNx4[float]],
    second_lines_projections_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> Tuple[float, float]:
    """
    Calculates repeatability and localization error
    :param first_lines_batch: list of lines for each first camera poses
    :param second_lines_batch: list of lines for each second camera poses
    :param first_lines_projections_batch: list of reprojected lines from first camera pose
    to corresponding second camera pose
    :param second_lines_projections_batch: list of reprojected lines from second camera pose
    to corresponding first camera pose
    :param distance: distance object or distance name used
    to determine correctly reprojected lines ({0})
    :param distance_threshold: threshold in pixels within which
    the line is considered to be correctly reprojected
    :return: repeatability and localization error
    """

    all_lines = list(
        it.chain(
            first_lines_batch,
            second_lines_batch,
            first_lines_projections_batch,
            second_lines_projections_batch,
        )
    )
    if any(
        np.max(lines) > EVALUATION_RESOLUTION if len(lines) != 0 else False
        for lines in all_lines
    ):
        raise ValueError(
            f"All lines should be scaled to the "
            f"{EVALUATION_RESOLUTION}x{EVALUATION_RESOLUTION} resolution"
        )

    distance = (
        DistanceFactory().from_string(distance)
        if isinstance(distance, str)
        else distance
    )

    def calculate_min_distances(first_lines, second_lines):
        min_distances = np.min(
            distance.calculate(
                first_lines,
                second_lines,
            ),
            axis=1,
        )
        return min_distances

    repeatability_sum = 0
    localization_error_sum = 0
    lines_shape = (-1, 2, 2)
    for (
        first_lines,
        second_lines,
        first_lines_projections,
        second_lines_projections,
    ) in zip(
        first_lines_batch,
        second_lines_batch,
        first_lines_projections_batch,
        second_lines_projections_batch,
    ):
        second_to_first_distances = calculate_min_distances(
            second_lines_projections.reshape(lines_shape),
            first_lines.reshape(lines_shape),
        )
        first_to_second_distances = calculate_min_distances(
            first_lines_projections.reshape(lines_shape),
            second_lines.reshape(lines_shape),
        )
        second_repeatable = second_to_first_distances < distance_threshold
        first_repeatable = first_to_second_distances < distance_threshold

        repeatability_sum += (first_repeatable.sum() + second_repeatable.sum()) / (
            len(first_lines) + len(second_lines)
        )
        second_to_first_localization_error = (
            0
            if second_repeatable.sum() == 0
            else second_to_first_distances[second_repeatable].mean()
        )
        first_to_second_localization_error = (
            0
            if first_repeatable.sum() == 0
            else first_to_second_distances[first_repeatable].mean()
        )
        localization_error_sum += (
            second_to_first_localization_error + first_to_second_localization_error
        ) / 2

    batch_size = len(first_lines_batch)

    repeatability_ = repeatability_sum / batch_size
    localization_error_ = localization_error_sum / batch_size

    return repeatability_, localization_error_


@docstring_arg(DISTANCE_NAMES)
def repeatability(
    first_lines_batch: List[ArrayNx4[float]],
    second_lines_batch: List[ArrayNx4[float]],
    first_lines_projections_batch: List[ArrayNx4[float]],
    second_lines_projections_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> float:
    """
    Calculates repeatability
    :param first_lines_batch: list of lines for each first camera poses
    :param second_lines_batch: list of lines for each second camera poses
    :param first_lines_projections_batch: list of reprojected lines from first camera pose
    to corresponding second camera pose
    :param second_lines_projections_batch: list of reprojected lines from second camera pose
    to corresponding first camera pose
    :param distance: distance object or distance name used
    to determine correctly reprojected lines ({0})
    :param distance_threshold: threshold in pixels within which
    the line is considered to be correctly reprojected
    :return: repeatability
    """

    repeatability_, _ = repeatability_localization_error(
        first_lines_batch,
        second_lines_batch,
        first_lines_projections_batch,
        second_lines_projections_batch,
        distance,
        distance_threshold,
    )
    return repeatability_


@docstring_arg(DISTANCE_NAMES)
def localization_error(
    first_lines_batch: List[ArrayNx4[float]],
    second_lines_batch: List[ArrayNx4[float]],
    first_lines_projections_batch: List[ArrayNx4[float]],
    second_lines_projections_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> float:
    """
    Calculates localization error
    :param first_lines_batch: list of lines for each first camera poses
    :param second_lines_batch: list of lines for each second camera poses
    :param first_lines_projections_batch: list of reprojected lines from first camera pose
    to corresponding second camera pose
    :param second_lines_projections_batch: list of reprojected lines from second camera pose
    to corresponding first camera pose
    :param distance: distance object or distance name used
    to determine correctly reprojected lines ({0})
    :param distance_threshold: threshold in pixels within which
    the line is considered to be correctly reprojected
    :return: localization error
    """

    _, localization_error_ = repeatability_localization_error(
        first_lines_batch,
        second_lines_batch,
        first_lines_projections_batch,
        second_lines_projections_batch,
        distance,
        distance_threshold,
    )
    return localization_error_
