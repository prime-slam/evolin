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

import os

import numpy as np
from joblib import Parallel, delayed
from tqdm.contrib import tzip
from typing import List, Tuple

from src.metrics.detection.heatmap import heatmap_precision_recall_curve
from src.metrics.detection.heatmap.tp_indicator import (
    HeatmapTPIndicator,
)
from src.metrics.detection.heatmap.utils import equally_sized, rasterize
from src.typing import ArrayNx4, ArrayN

__all__ = [
    "heatmap_recall",
    "heatmap_precision",
    "heatmap_precision_recall_fscore",
    "heatmap_max_fscore",
    "heatmap_fscore",
]


class PrecisionRecall:
    """
    Class that calculates heatmap precision and recall
    over batches of predicted and ground truth lines
    """

    def __init__(self):
        self.indicator = HeatmapTPIndicator()

    def calculate(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
        heights_batch: ArrayN[int],
        widths_batch: ArrayN[int],
    ) -> Tuple[float, float]:
        """
        Calculates heatmap precision and recall
        :param pred_lines_batch: list of predicted lines for each image
        :param gt_lines_batch: list of ground truth lines for each image
        :param heights_batch: array of heights of each image
        :param widths_batch: array of widths of each image
        :return: heatmap precision and recall values
        """
        if not equally_sized(
            [
                pred_lines_batch,
                gt_lines_batch,
                heights_batch,
                widths_batch,
            ]
        ):
            raise ValueError("All batches must be the same size")

        tp = 0
        fp = 0
        gt_size = 0

        def add_statistics(pred_lines, gt_lines, height, width):
            nonlocal tp, fp, gt_size
            gt_map = rasterize(gt_lines, height, width)
            pred_map = rasterize(pred_lines, height, width)
            tp_indicators_map = self.indicator.indicate(gt_map, pred_map)
            tp += tp_indicators_map.nnz
            fp += pred_map.nnz - tp_indicators_map.nnz
            gt_size += gt_map.nnz

        Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
            delayed(add_statistics)(pred_lines, gt_lines, height, width)
            for pred_lines, gt_lines, height, width in tzip(
                pred_lines_batch,
                gt_lines_batch,
                heights_batch,
                widths_batch,
            )
        )

        recall = tp / gt_size
        precision = tp / (tp + fp) if tp + fp != 0 else 0

        return precision, recall


def heatmap_precision_recall_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
) -> Tuple[float, float, float]:
    """
    Calculates heatmap precision and recall
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param heights_batch: array of heights of each image
    :param widths_batch: array of widths of each image
    :return: heatmap precision and recall values
    """
    precision, recall = PrecisionRecall().calculate(
        pred_lines_batch, gt_lines_batch, heights_batch, widths_batch
    )
    fscore = (
        2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    )

    return precision, recall, fscore


def heatmap_precision(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
) -> float:
    """
    Calculates heatmap precision
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param heights_batch: array of heights of each image
    :param widths_batch: array of widths of each image
    :return: heatmap precision
    """
    precision, _ = PrecisionRecall().calculate(
        pred_lines_batch, gt_lines_batch, heights_batch, widths_batch
    )

    return precision


def heatmap_recall(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
) -> float:
    """
    Calculates heatmap recall
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param heights_batch: array of heights of each image
    :param widths_batch: array of widths of each image
    :return: heatmap recall
    """
    _, recall = PrecisionRecall().calculate(
        pred_lines_batch, gt_lines_batch, heights_batch, widths_batch
    )

    return recall


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
    _, _, fscore = heatmap_precision_recall_fscore(
        pred_lines_batch, gt_lines_batch, heights_batch, widths_batch
    )

    return fscore


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
