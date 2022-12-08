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

__all__ = ["heatmap_average_precision"]


class HeatmapAveragePrecision:
    """
    Class that calculates the Heatmap Average Precision
    over batches of predicted and ground truth lines
    """

    def __init__(self):
        self.precision_recall_calculator = HeatmapPrecisionRecall()

    def calculate(
        self,
        pred_lines_batch: List[ArrayNx4[np.float]],
        gt_lines_batch: List[ArrayNx4[np.float]],
        scores_batch: List[ArrayN[np.float]],
        heights_batch: ArrayN[np.int],
        widths_batch: ArrayN[np.int],
        thresholds: ArrayN[np.int],
    ):
        precision, recall = self.precision_recall_calculator.calculate(
            pred_lines_batch,
            gt_lines_batch,
            scores_batch,
            heights_batch,
            widths_batch,
            thresholds,
        )

        order = np.argsort(recall)
        return np.trapz(x=recall[order], y=precision[order])


def heatmap_average_precision(
    pred_lines_batch: List[ArrayNx4[np.float]],
    gt_lines_batch: List[ArrayNx4[np.float]],
    line_scores_batch: List[ArrayNx4[np.float]],
    heights_batch: ArrayN[np.int],
    widths_batch: ArrayN[np.int],
    thresholds: ArrayN[np.int],
):
    """
    Calculates Heatmap Average Precision (AP^H)
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param line_scores_batch: list of predicted lines scores for each image
    :param heights_batch: array of heights of each image
    :param widths_batch: array of widths of each image
    :param thresholds: array of line scores thresholds to filter predicted lines
    :return: Heatmap Average Precision value
    """
    return HeatmapAveragePrecision().calculate(
        pred_lines_batch,
        gt_lines_batch,
        line_scores_batch,
        heights_batch,
        widths_batch,
        thresholds,
    )
