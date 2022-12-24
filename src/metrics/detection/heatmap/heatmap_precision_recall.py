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
import os

from joblib import Parallel, delayed
from tqdm.contrib import tzip
from typing import List, Tuple

from src.metrics.detection.heatmap.assignment_problem.heatmap_tp_indicator import (
    HeatmapTPIndicator,
)
from src.metrics.detection.heatmap.utils import equally_sized, rasterize
from src.typing import ArrayNx4, ArrayN

__all__ = ["HeatmapPrecisionRecall"]


class HeatmapPrecisionRecall:
    """
    Class that calculates Heatmap Precision and Recall
    over batches of predicted and ground truth lines
    """

    def __init__(self):
        self.indicator = HeatmapTPIndicator()
        self.epsilon = 1e-9

    def calculate(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
        scores_batch: List[ArrayNx4[float]],
        heights_batch: ArrayN[int],
        widths_batch: ArrayN[int],
        thresholds: ArrayN[int],
    ) -> Tuple[ArrayN[float], ArrayN[float]]:
        if not equally_sized(
            [
                pred_lines_batch,
                gt_lines_batch,
                scores_batch,
                heights_batch,
                widths_batch,
            ]
        ):
            raise ValueError("All batches must be the same size")

        thresholds_number = len(thresholds)
        if thresholds_number == 0:
            raise ValueError("The list of threshold cannot be empty")

        tp_sum = np.zeros(thresholds_number)
        fp_sum = np.zeros(thresholds_number)
        gt_size_sum = np.zeros(thresholds_number)

        def add_statistics(pred_lines, gt_lines, scores, height, width):
            for i, threshold in enumerate(thresholds):
                gt_map = rasterize(gt_lines, height, width)
                pred_map = rasterize(pred_lines[scores > threshold], height, width)
                tp_indicators_map = self.indicator.indicate(gt_map, pred_map)
                tp_sum[i] += np.count_nonzero(tp_indicators_map)
                fp_sum[i] += np.count_nonzero(pred_map) - tp_sum[i]
                gt_size_sum[i] += np.count_nonzero(gt_map)

        Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
            delayed(add_statistics)(pred_lines, gt_lines, scores, height, width)
            for pred_lines, gt_lines, scores, height, width in tzip(
                pred_lines_batch,
                gt_lines_batch,
                scores_batch,
                heights_batch,
                widths_batch,
            )
        )

        recall = tp_sum / gt_size_sum
        precision = tp_sum / np.maximum(tp_sum + fp_sum, self.epsilon)

        return precision, recall
