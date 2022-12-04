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
from typing import List

from src.metrics.detection.structural.average_precision import average_precision
from src.metrics.detection.structural.distance.tp_indicator import (
    __calculate_orthogonal_tp_indicators,
)

__all__ = ["orthogonal_average_precision"]


def orthogonal_average_precision(
    pred_lines_batch: List[np.ndarray],
    gt_lines_batch: List[np.ndarray],
    line_scores_batch: List[np.ndarray],
    distance_threshold=5,
    min_overlap=0.5,
):
    """
    Calculates Orthogonal Average Precision (OAP)
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param line_scores_batch: list of predicted lines scores for each image
    :param distance_threshold: threshold in pixels within which the line is considered to be true positive
    :param min_overlap: minimal overlap of the projection of one line onto another line, averaged over two lines;
    lines with a value greater than the threshold to be true positive
    :return: Orthogonal Average Precision value
    """
    orthogonal_tp_indication = partial(
        __calculate_orthogonal_tp_indicators,
        distance_threshold=distance_threshold,
        min_overlap=min_overlap,
    )
    return average_precision(
        pred_lines_batch,
        gt_lines_batch,
        line_scores_batch,
        tp_indication=orthogonal_tp_indication,
    )
