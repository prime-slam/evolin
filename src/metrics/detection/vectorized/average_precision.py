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

from typing import List, Union

from src.metrics.detection.vectorized.constants import DISTANCE_NAMES
from src.metrics.detection.vectorized.distance.distance import Distance
from src.metrics.detection.vectorized.distance.distance_factory import (
    DistanceFactory,
)
from src.metrics.detection.vectorized.precision_recall_curve import PrecisionRecallCurve
from src.metrics.detection.vectorized.utils import docstring_arg
from src.typing import ArrayNx4, ArrayN
from src.metrics.detection.vectorized.tp_indicator import (
    VectorizedTPIndicator,
)

__all__ = [
    "vectorized_average_precision",
]


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
        :param tp_indicator: VectorizedTPIndicator object that indicates
        whether line is true positive or not
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
        precision, recall = PrecisionRecallCurve(self.tp_indicator).calculate(
            pred_lines_batch, gt_lines_batch, line_scores_batch
        )
        # AP is the area under the PR Curve
        return np.trapz(x=recall, y=precision)


@docstring_arg(DISTANCE_NAMES)
def vectorized_average_precision(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    line_scores_batch: List[ArrayN[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> float:
    """
    Calculates vectorized average precision
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param line_scores_batch: list of predicted lines scores for each image
    :param distance: distance object or distance name used
    to determine true positives ({0})
    :param distance_threshold: threshold in pixels within which
    the line is considered to be true positive
    :return: vectorized average precision value
    """

    distance = (
        DistanceFactory().from_string(distance)
        if isinstance(distance, str)
        else distance
    )

    tp_indicator = VectorizedTPIndicator(distance, distance_threshold)

    return AveragePrecision(tp_indicator=tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
        line_scores_batch,
    )
