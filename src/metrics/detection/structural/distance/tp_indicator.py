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

from functools import partial
from typing import Callable

from src.metrics.detection.structural.constants import EVALUATION_RESOLUTION
from src.metrics.detection.structural.distance.orthogonal import (
    __calculate_averaged_orthogonal_distance,
)
from src.metrics.detection.structural.distance.structural import (
    __calculate_structural_distance,
)


def __calculate_structural_tp_indicators(
    lines_pred: np.ndarray,
    lines_gt: np.ndarray,
    distance_threshold: float = 5,
) -> np.ndarray:
    return __calculate_tp_indicators(
        lines_pred,
        lines_gt,
        distance_threshold,
        distance=__calculate_structural_distance,
    )


def __calculate_orthogonal_tp_indicators(
    lines_pred: np.ndarray,
    lines_gt: np.ndarray,
    distance_threshold: float = 5,
    min_overlap: float = 0.5,
) -> np.ndarray:
    orthogonal_distance = partial(
        __calculate_averaged_orthogonal_distance, min_overlap=min_overlap
    )
    return __calculate_tp_indicators(
        lines_pred, lines_gt, distance_threshold, distance=orthogonal_distance
    )


def __calculate_tp_indicators(
    lines_pred: np.ndarray,
    lines_gt: np.ndarray,
    distance_threshold: float,
    distance: Callable,
) -> np.ndarray:
    if (
        np.max(lines_pred) > EVALUATION_RESOLUTION
        or np.max(lines_gt) > EVALUATION_RESOLUTION
    ):
        raise ValueError(
            f"The detection results and the ground truth "
            f"lines should be scaled to the "
            f"{EVALUATION_RESOLUTION}x{EVALUATION_RESOLUTION} resolution"
        )

    # [x1, y1, x2, y2] -> [[x1, y1], [x2, y2]]
    lines_pred = lines_pred.reshape((-1, 2, 2))
    lines_gt = lines_gt.reshape((-1, 2, 2))

    if np.any(np.logical_and.reduce(lines_pred[:, 0] == lines_pred[:, 1])):
        raise ValueError(
            "The segment of zero length is contained in the set of predicted lines"
        )

    if np.any(np.logical_and.reduce(lines_gt[:, 0] == lines_gt[:, 1])):
        raise ValueError(
            "The segment of zero length is contained in the set of gt lines"
        )

    distances = distance(lines_pred, lines_gt)

    closest_gt_lines_indices = np.argmin(distances, 1)
    closest_gt_lines_distances = distances[
        np.arange(distances.shape[0]), closest_gt_lines_indices
    ]

    predictions_number = len(lines_pred)
    hit = np.zeros(len(lines_gt), bool)
    tp = np.zeros(predictions_number, bool)

    for pred_line in range(predictions_number):
        if (
            closest_gt_lines_distances[pred_line] < distance_threshold
            and not hit[closest_gt_lines_indices[pred_line]]
        ):
            hit[closest_gt_lines_indices[pred_line]] = True
            tp[pred_line] = True
    return tp
