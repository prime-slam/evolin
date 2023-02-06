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

from typing import List, Tuple, Union

from src.metrics.detection.vectorized.constants import DISTANCE_NAMES
from src.metrics.detection.vectorized.distance.distance import Distance
from src.metrics.detection.vectorized.distance.distance_factory import (
    DistanceFactory,
)
from src.metrics.detection.vectorized.utils import docstring_arg
from src.typing import ArrayNx4, ArrayN
from src.metrics.detection.vectorized.tp_indicator import (
    VectorizedTPIndicator,
)

__all__ = [
    "vectorized_precision_recall_curve",
]


class PrecisionRecallCurve:
    """
    Class that calculates precision-recall pairs for different score thresholds
    over batches of predicted and ground truth lines
    """

    def __init__(
        self,
        tp_indicator: VectorizedTPIndicator,
    ):
        """
        :param tp_indicator: VectorizedTPIndicator object that indicates
        whether line is true positive or not
        """
        self.tp_indicator = tp_indicator

    def calculate(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
        line_scores_batch: List[ArrayN[float]],
    ) -> Tuple[ArrayN[float], ArrayN[float]]:
        """
        Calculates the Precision-Recall Curve
        :param pred_lines_batch: list of predicted lines for each image
        :param gt_lines_batch: list of ground truth lines for each image
        :param line_scores_batch: list of predicted lines scores for each image
        :return: lists of x (recall) and y (precision) coordinates
        """

        total_tp_indicators = []
        sorted_scores = []
        for pred_lines, gt_lines, scores in zip(
            pred_lines_batch, gt_lines_batch, line_scores_batch
        ):
            score_descending_order = np.argsort(-scores)
            total_tp_indicators.append(
                self.tp_indicator.indicate(pred_lines[score_descending_order], gt_lines)
            )
            sorted_scores.append(scores[score_descending_order])

        sorted_scores = np.concatenate(sorted_scores)
        total_tp_indicators = np.concatenate(total_tp_indicators)
        total_fp_indicators = ~total_tp_indicators
        gt_size = sum(len(gt_lines) for gt_lines in gt_lines_batch)

        precision_descending_order = np.argsort(-sorted_scores)
        total_tp_indicators = total_tp_indicators[precision_descending_order]
        total_fp_indicators = total_fp_indicators[precision_descending_order]

        tp = np.cumsum(total_tp_indicators)
        fp = np.cumsum(total_fp_indicators)

        recall = tp / gt_size
        precision = np.zeros(np.size(tp), dtype=float)
        nonzero_mask = tp + fp != 0
        precision[nonzero_mask] = tp[nonzero_mask] / (
            tp[nonzero_mask] + fp[nonzero_mask]
        )

        return precision, recall


@docstring_arg(DISTANCE_NAMES)
def vectorized_precision_recall_curve(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    line_scores_batch: List[ArrayN[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> Tuple[ArrayN[float], ArrayN[float]]:
    """
    Calculates the Orthogonal Precision-Recall Curve
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param line_scores_batch: list of predicted lines scores for each image
    :param distance: distance object or distance name used
    to determine true positives ({0})
    :param distance_threshold: threshold in pixels within which the line is considered to be true positive
    :return: lists of x (recall) and y (precision) coordinates
    """
    distance = (
        DistanceFactory().from_string(distance)
        if isinstance(distance, str)
        else distance
    )

    tp_indicator = VectorizedTPIndicator(distance, distance_threshold)

    return PrecisionRecallCurve(tp_indicator=tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
        line_scores_batch,
    )
