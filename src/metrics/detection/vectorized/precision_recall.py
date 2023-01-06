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

from typing import List, Tuple

from src.metrics.detection.vectorized.distance.orthogonal import OrthogonalDistance
from src.metrics.detection.vectorized.distance.structural import StructuralDistance
from src.typing import ArrayNx4
from src.metrics.detection.vectorized.tp_indicator import (
    VectorizedTPIndicator,
)

__all__ = [
    "orthogonal_recall",
    "structural_recall",
    "orthogonal_precision",
    "structural_precision",
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


def orthogonal_precision_recall(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance_threshold: float = 5,
    min_overlap: float = 0.5,
) -> Tuple[float, float]:
    """
    Calculates orthogonal precision and recall
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance_threshold: threshold in pixels within which the line is considered to be true positive
    :param min_overlap: minimal overlap of the projection of one line onto another line, averaged over two lines;
    lines with a value greater than the threshold to be true positive
    :return: orthogonal precision and recall values
    """
    orthogonal_tp_indicator = VectorizedTPIndicator(
        OrthogonalDistance(min_overlap), distance_threshold
    )
    precision, recall = PrecisionRecall(tp_indicator=orthogonal_tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
    )

    return precision, recall


def structural_precision_recall(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance_threshold: float = 5,
) -> Tuple[float, float]:
    """
    Calculates structural precision and recall
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance_threshold: threshold in pixels within which the line is considered to be true positive
    :return: structural precision and recall values
    """
    structural_tp_indicator = VectorizedTPIndicator(
        StructuralDistance(), distance_threshold
    )
    precision, recall = PrecisionRecall(tp_indicator=structural_tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
    )

    return precision, recall


def orthogonal_precision(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance_threshold: float = 5,
    min_overlap: float = 0.5,
) -> float:
    """
    Calculates orthogonal precision
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance_threshold: threshold in pixels within which the line is considered to be true positive
    :param min_overlap: minimal overlap of the projection of one line onto another line, averaged over two lines;
    lines with a value greater than the threshold to be true positive
    :return: orthogonal precision value
    """
    orthogonal_tp_indicator = VectorizedTPIndicator(
        OrthogonalDistance(min_overlap), distance_threshold
    )
    precision, _ = PrecisionRecall(tp_indicator=orthogonal_tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
    )

    return precision


def structural_precision(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance_threshold: float = 5,
) -> float:
    """
    Calculates structural precision
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance_threshold: threshold in pixels within which the line is considered to be true positive
    :return: structural precision value
    """

    structural_tp_indicator = VectorizedTPIndicator(
        StructuralDistance(), distance_threshold
    )

    precision, _ = PrecisionRecall(tp_indicator=structural_tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
    )

    return precision


def orthogonal_recall(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance_threshold: float = 5,
    min_overlap: float = 0.5,
) -> float:
    """
    Calculates orthogonal recall
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance_threshold: threshold in pixels within which the line is considered to be true positive
    :param min_overlap: minimal overlap of the projection of one line onto another line, averaged over two lines;
    lines with a value greater than the threshold to be true positive
    :return: orthogonal recall value
    """
    orthogonal_tp_indicator = VectorizedTPIndicator(
        OrthogonalDistance(min_overlap), distance_threshold
    )

    _, recall = PrecisionRecall(tp_indicator=orthogonal_tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
    )

    return recall


def structural_recall(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance_threshold: float = 5,
) -> float:
    """
    Calculates structural recall
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance_threshold: threshold in pixels within which the line is considered to be true positive
    :return: structural recall value
    """

    structural_tp_indicator = VectorizedTPIndicator(
        StructuralDistance(), distance_threshold
    )

    _, recall = PrecisionRecall(tp_indicator=structural_tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
    )

    return recall
