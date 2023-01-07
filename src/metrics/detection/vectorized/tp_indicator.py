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

from src.metrics.detection.vectorized.constants import EVALUATION_RESOLUTION
from src.metrics.detection.vectorized.distance.distance import Distance
from src.metrics.detection.vectorized.utils import contains_zero_length_line
from src.typing import ArrayNx4, ArrayN


class VectorizedTPIndicator:
    def __init__(self, distance: Distance, distance_threshold: float):
        self.distance = distance
        self.distance_threshold = distance_threshold

    def indicate(
        self,
        pred_lines: ArrayNx4[float],
        gt_lines: ArrayNx4[float],
        sort_predictions_by_distance: bool = False,
    ) -> ArrayN[bool]:
        if (
            np.max(pred_lines) > EVALUATION_RESOLUTION
            or np.max(gt_lines) > EVALUATION_RESOLUTION
        ):
            raise ValueError(
                f"The detection results and the ground truth "
                f"lines should be scaled to the "
                f"{EVALUATION_RESOLUTION}x{EVALUATION_RESOLUTION} resolution"
            )

        # [x1, y1, x2, y2] -> [[x1, y1], [x2, y2]]
        pred_lines = pred_lines.reshape((-1, 2, 2))
        gt_lines = gt_lines.reshape((-1, 2, 2))

        if contains_zero_length_line(pred_lines):
            raise ValueError(
                "The segment of zero length is contained in the set of predicted lines"
            )

        if contains_zero_length_line(gt_lines):
            raise ValueError(
                "The segment of zero length is contained in the set of gt lines"
            )

        distances = self.distance.calculate(pred_lines, gt_lines)

        closest_gt_lines_indices = np.argmin(distances, axis=1)
        closest_gt_lines_distances = distances[
            np.arange(distances.shape[0]), closest_gt_lines_indices
        ]

        predictions_number = len(pred_lines)
        hit = np.zeros(len(gt_lines), bool)
        tp = np.zeros(predictions_number, bool)

        prediction_indices = (
            np.argsort(closest_gt_lines_distances)
            if sort_predictions_by_distance
            else np.arange(predictions_number)
        )

        for pred_line in prediction_indices:
            if (
                closest_gt_lines_distances[pred_line] < self.distance_threshold
                and not hit[closest_gt_lines_indices[pred_line]]
            ):
                hit[closest_gt_lines_indices[pred_line]] = True
                tp[pred_line] = True

        return tp
