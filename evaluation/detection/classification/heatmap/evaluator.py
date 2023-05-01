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

from evaluation.common.evaluator_base import Evaluator
from evaluation.common.metric_information.metric_info import (
    MetricInfo,
    SimpleMetricInfo,
    CompositeMetricInfo,
)
from evolin.metrics.detection.heatmap import (
    heatmap_precision_recall_fscore,
    heatmap_precision_recall_curve,
)
from evolin.typing import ArrayNx4, ArrayN


class UnscoredEvaluator(Evaluator):
    def __init__(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
        heights_batch: ArrayN[int],
        widths_batch: ArrayN[int],
    ):
        self.pred_lines_batch = pred_lines_batch
        self.gt_lines_batch = gt_lines_batch
        self.heights_batch = heights_batch
        self.widths_batch = widths_batch

    def evaluate(self) -> List[MetricInfo]:
        precision, recall, fscore = heatmap_precision_recall_fscore(
            self.pred_lines_batch,
            self.gt_lines_batch,
            self.heights_batch,
            self.widths_batch,
        )
        return [
            SimpleMetricInfo(
                name="precision", value=precision, additional_information=None
            ),
            SimpleMetricInfo(name="recall", value=recall, additional_information=None),
            SimpleMetricInfo(name="fscore", value=fscore, additional_information=None),
        ]


class ScoredEvaluator(Evaluator):
    def __init__(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
        scores_batch: List[ArrayN[float]],
        heights_batch: ArrayN[int],
        widths_batch: ArrayN[int],
        score_thresholds: ArrayN[float],
    ):
        self.pred_lines_batch = pred_lines_batch
        self.gt_lines_batch = gt_lines_batch
        self.scores_batch = scores_batch
        self.score_thresholds = score_thresholds
        self.heights_batch = heights_batch
        self.widths_batch = widths_batch

    def evaluate(self) -> List[MetricInfo]:
        precision, recall = heatmap_precision_recall_curve(
            self.pred_lines_batch,
            self.gt_lines_batch,
            self.scores_batch,
            self.heights_batch,
            self.widths_batch,
            self.score_thresholds,
        )

        fscore = np.zeros(precision.size, dtype=float)
        nonzero_mask = precision + recall != 0
        fscore[nonzero_mask] = (
            2
            * precision[nonzero_mask]
            * recall[nonzero_mask]
            / (precision[nonzero_mask] + recall[nonzero_mask])
        )

        max_fscore = np.max(fscore)
        average_precision = np.trapz(x=recall, y=precision)
        return [
            CompositeMetricInfo(
                name="pr_curve",
                values={"x": recall.tolist(), "y": precision.tolist()},
                additional_information=None,
            ),
            SimpleMetricInfo(
                name="average_precision",
                value=average_precision,
                additional_information=None,
            ),
            SimpleMetricInfo(
                name="max_fscore", value=max_fscore, additional_information=None
            ),
        ]
