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

from src.typing import ArrayNx4, ArrayN
from src.metrics.detection.vectorized.distance.vectorized_tp_indicator import (
    VectorizedTPIndicator,
)

__all__ = ["AveragePrecision"]


class AveragePrecision:
    """
    Class that calculates the Average Precision
    over batches of predicted and ground truth lines
    """

    def __init__(
        self,
        tp_indicator: VectorizedTPIndicator,
    ):
        """
        :param tp_indicator: TPIndicator object that indicates whether line is true positive or not
        """
        self.tp_indicator = tp_indicator

    def calculate(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
        line_scores_batch: List[ArrayN[float]],
    ) -> float:
        """
        Calculates Average Precision
        :param pred_lines_batch: list of predicted lines for each image
        :param gt_lines_batch: list of ground truth lines for each image
        :param line_scores_batch: list of predicted lines scores for each image
        :return: Average Precision value
        """
        total_tp_indicators = [
            # sort pred_lines in descending order of scores
            self.tp_indicator.indicate(pred_lines[np.argsort(-scores)], gt_lines)
            for pred_lines, gt_lines, scores in zip(
                pred_lines_batch, gt_lines_batch, line_scores_batch
            )
        ]

        total_tp_indicators = np.concatenate(total_tp_indicators, dtype=bool)
        total_fp_indicators = ~total_tp_indicators
        total_scores = np.concatenate(line_scores_batch)
        gt_size = sum(len(gt_lines) for gt_lines in gt_lines_batch)

        index = np.argsort(-total_scores)
        tp = np.cumsum(total_tp_indicators[index])
        fp = np.cumsum(total_fp_indicators[index])

        epsilon = 1e-9
        recall = tp / gt_size  # gt_size = tp + fn
        precision = tp / np.maximum(tp + fp, epsilon)

        return np.trapz(x=recall, y=precision)
