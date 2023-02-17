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
from evaluation.common.metric_information.additional_info import (
    DistanceInfo,
    ScoreInfo,
)
from evaluation.common.metric_information.metric_info import (
    MetricInfo,
    SimpleMetricInfo,
    CompositeMetricInfo,
)
from evaluation.common.utils import filter_lines_by_score
from src.metrics.detection.vectorized import (
    vectorized_precision_recall_curve,
    vectorized_precision_recall_fscore,
)
from src.metrics.detection.vectorized.distance.distance_factory import DistanceName
from src.typing import ArrayNx4, ArrayN


class UnscoredEvaluator(Evaluator):
    def __init__(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
        distance_thresholds: ArrayN[float],
    ):
        self.pred_lines_batch = pred_lines_batch
        self.gt_lines_batch = gt_lines_batch
        self.distance_thresholds = distance_thresholds

    def evaluate(self) -> List[MetricInfo]:
        results = []
        for dist in DistanceName:
            dist_name = dist.name
            for dist_threshold in self.distance_thresholds:
                precision, recall, fscore = vectorized_precision_recall_fscore(
                    self.pred_lines_batch,
                    self.gt_lines_batch,
                    distance=dist_name,
                    distance_threshold=dist_threshold,
                )
                dist_info = DistanceInfo(
                    distance=dist_name, distance_threshold=dist_threshold
                )
                results.extend(
                    [
                        SimpleMetricInfo(
                            name="precision",
                            value=precision,
                            additional_information=[dist_info],
                        ),
                        SimpleMetricInfo(
                            name="recall",
                            value=recall,
                            additional_information=[dist_info],
                        ),
                        SimpleMetricInfo(
                            name="fscore",
                            value=fscore,
                            additional_information=[dist_info],
                        ),
                    ]
                )
        return results


class ScoredEvaluator(Evaluator):
    def __init__(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
        scores_batch: List[ArrayN[float]],
        score_thresholds: ArrayN[float],
        distance_thresholds: ArrayN[float],
    ):
        self.pred_lines_batch = pred_lines_batch
        self.gt_lines_batch = gt_lines_batch
        self.scores_batch = scores_batch
        self.score_thresholds = score_thresholds
        self.distance_thresholds = distance_thresholds

    def evaluate(self) -> List[MetricInfo]:
        results = []
        for dist in DistanceName:
            dist_name = dist.name
            for dist_threshold in self.distance_thresholds:
                precision, recall = vectorized_precision_recall_curve(
                    self.pred_lines_batch,
                    self.gt_lines_batch,
                    self.scores_batch,
                    distance=dist_name,
                    distance_threshold=dist_threshold,
                )
                average_precision = np.trapz(x=recall, y=precision)
                dist_info = DistanceInfo(
                    distance=dist_name, distance_threshold=dist_threshold
                )
                results.extend(
                    [
                        SimpleMetricInfo(
                            name="average_precision",
                            value=average_precision,
                            additional_information=[dist_info],
                        ),
                        CompositeMetricInfo(
                            name="pr_curve",
                            values={"x": recall.tolist(), "y": precision.tolist()},
                            additional_information=[dist_info],
                        ),
                    ]
                )

        for score_threshold in self.score_thresholds:
            filtered_lines_batch = filter_lines_by_score(
                self.pred_lines_batch, self.scores_batch, score_threshold
            )
            metrics_info = UnscoredEvaluator(
                filtered_lines_batch,
                self.gt_lines_batch,
                self.distance_thresholds,
            ).evaluate()

            for info in metrics_info:
                info.additional_information.append(
                    ScoreInfo(score_threshold=score_threshold)
                )
            results.extend(metrics_info)

        return results
