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

from src.metrics.detection.heatmap.heatmap_precision_recall import (
    HeatmapPrecisionRecall,
)
from src.typing import ArrayNx4, ArrayN

__all__ = ["heatmap_fscore"]


class HeatmapFScore:
    """
    Class that calculates F-Score
    over batches of predicted and ground truth lines
    """

    def __init__(self):
        self.precision_recall_calculator = HeatmapPrecisionRecall()
        self.epsilon = 1e-9

    def calculate(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
        scores_batch: List[ArrayNx4[float]],
        heights_batch: ArrayN[int],
        widths_batch: ArrayN[int],
        thresholds: ArrayN[int],
    ):
        recall, precision = self.precision_recall_calculator.calculate(
            pred_lines_batch,
            gt_lines_batch,
            scores_batch,
            heights_batch,
            widths_batch,
            thresholds,
        )

        fscore = 2 * precision * recall / np.maximum(precision + recall, self.epsilon)

        return np.max(fscore)


def heatmap_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    line_scores_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
    thresholds: ArrayN[int],
):
    """
    Calculates Heatmap F-Score (F^H)
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param line_scores_batch: list of predicted lines scores for each image
    :param heights_batch: array of heights of each image
    :param widths_batch: array of widths of each image
    :param thresholds: array of line scores thresholds to filter predicted lines
    :return: Heatmap F-Score value
    """
    return HeatmapFScore().calculate(
        pred_lines_batch,
        gt_lines_batch,
        line_scores_batch,
        heights_batch,
        widths_batch,
        thresholds,
    )
