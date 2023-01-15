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

from typing import List, Tuple, Union

from src.metrics.detection.vectorized import DISTANCE_NAMES
from src.metrics.detection.vectorized.distance.distance import Distance
from src.metrics.detection.vectorized.distance.distance_factory import (
    DistanceFactory,
)
from src.metrics.detection.vectorized.utils import docstring_arg
from src.typing import ArrayNx4
from src.metrics.detection.vectorized.tp_indicator import (
    VectorizedTPIndicator,
)

__all__ = [
    "vectorized_precision_recall_fscore",
    "vectorized_precision",
    "vectorized_recall",
    "vectorized_fscore",
]


class PrecisionRecall:
    """
    Class that calculates precision and recall
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
    ) -> Tuple[float, float]:
        """
        Calculates precision and recall
        :param pred_lines_batch: list of predicted lines for each image
        :param gt_lines_batch: list of ground truth lines for each image
        :return: precision and recall values
        """

        tp = sum(
            self.tp_indicator.indicate(
                pred_lines, gt_lines, sort_predictions_by_distance=True
            ).sum()
            for pred_lines, gt_lines in zip(pred_lines_batch, gt_lines_batch)
        )
        gt_size = sum(len(gt_lines) for gt_lines in gt_lines_batch)
        pred_size = sum(len(pred_lines) for pred_lines in pred_lines_batch)
        fp = pred_size - tp

        recall = tp / gt_size
        precision = tp / (tp + fp) if tp + fp != 0 else 0

        return precision, recall


@docstring_arg(DISTANCE_NAMES)
def vectorized_precision_recall_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> Tuple[float, float, float]:
    """
    Calculates vectorized precision and recall
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance: distance object or distance name used
    to determine true positives ({0})
    :param distance_threshold: threshold in pixels within which
    the line is considered to be true positive
    :return: vectorized precision and recall values
    """

    distance = (
        DistanceFactory().from_string(distance)
        if isinstance(distance, str)
        else distance
    )

    tp_indicator = VectorizedTPIndicator(distance, distance_threshold)

    precision, recall = PrecisionRecall(tp_indicator=tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
    )
    fscore = (
        2 * precision * recall / (precision + recall) if precision * recall != 0 else 0
    )

    return precision, recall, fscore


@docstring_arg(DISTANCE_NAMES)
def vectorized_precision(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> float:
    """
    Calculates vectorized precision
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance: distance object or distance name used
    to determine true positives ({0})
    :param distance_threshold: threshold in pixels within which
    the line is considered to be true positive
    :return: vectorized precision value
    """

    precision, _, _ = vectorized_precision_recall_fscore(
        pred_lines_batch, gt_lines_batch, distance, distance_threshold
    )

    return precision


@docstring_arg(DISTANCE_NAMES)
def vectorized_recall(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> float:
    """
    Calculates vectorized recall
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance: distance object or distance name used
    to determine true positives ({0})
    :param distance_threshold: threshold in pixels within which
    the line is considered to be true positive
    :return: vectorized recall value
    """

    _, recall, _ = vectorized_precision_recall_fscore(
        pred_lines_batch, gt_lines_batch, distance, distance_threshold
    )

    return recall


@docstring_arg(DISTANCE_NAMES)
def vectorized_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> float:
    """
    Calculates vectorized F-Score
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance: distance object or distance name used
    to determine true positives ({0})
    :param distance_threshold: threshold in pixels within which
    the line is considered to be true positive
    :return: vectorized F-Score value
    """
    _, _, fscore = vectorized_precision_recall_fscore(
        pred_lines_batch,
        gt_lines_batch,
        distance,
        distance_threshold,
    )

    return fscore
