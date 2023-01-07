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

from src.metrics.detection.heatmap.precision_recall import heatmap_precision_recall
from src.metrics.detection.heatmap.precision_recall_curve import (
    heatmap_precision_recall_curve,
)
from src.typing import ArrayNx4, ArrayN

__all__ = ["heatmap_max_fscore", "heatmap_fscore"]


def heatmap_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
):
    """
    Calculates heatmap F-Score
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param heights_batch: array of heights of each image
    :param widths_batch: array of widths of each image
    :return: heatmap F-Score value
    """
    precision, recall = heatmap_precision_recall(
        pred_lines_batch, gt_lines_batch, heights_batch, widths_batch
    )

    return (
        2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    )


def heatmap_max_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    line_scores_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
    thresholds: ArrayN[int],
):
    """
    Calculates the maximum F-Score among all thresholds
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param line_scores_batch: list of predicted lines scores for each image
    :param heights_batch: array of heights of each image
    :param widths_batch: array of widths of each image
    :param thresholds: array of line scores thresholds to filter predicted lines
    :return: maximum F-Score value
    """
    precision, recall = heatmap_precision_recall_curve(
        pred_lines_batch,
        gt_lines_batch,
        line_scores_batch,
        heights_batch,
        widths_batch,
        thresholds,
    )

    fscore = np.zeros(precision.size, dtype=float)
    nonzero_mask = precision + recall != 0
    fscore[nonzero_mask] = (
        2
        * precision[nonzero_mask]
        * recall[nonzero_mask]
        / (precision[nonzero_mask] + recall[nonzero_mask])
    )

    return np.max(fscore)
