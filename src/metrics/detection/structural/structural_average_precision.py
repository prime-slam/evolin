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

from typing import List

from src.metrics.detection.structural.average_precision import AveragePrecision
from src.metrics.detection.structural.distance.structural import StructuralDistance
from src.metrics.detection.structural.distance.tp_indicator import TPIndicator

__all__ = ["structural_average_precision"]


def structural_average_precision(
    pred_lines_batch: List[np.ndarray],
    gt_lines_batch: List[np.ndarray],
    line_scores_batch: List[np.ndarray],
    distance_threshold: float = 5,
) -> float:
    """
    Calculates Structural Average Precision (SAP)
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param line_scores_batch: list of predicted lines scores for each image
    :param distance_threshold: threshold in pixels within which the line is considered to be true positive
    :return: Structural Average Precision value
    """

    structural_tp_indicator = TPIndicator(StructuralDistance(), distance_threshold)

    return AveragePrecision(tp_indicator=structural_tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
        line_scores_batch,
    )
