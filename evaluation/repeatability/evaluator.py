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

from pathlib import Path
from typing import List

from evaluation.common.evaluator_base import Evaluator
from evaluation.common.metric_information.additional_info import (
    DistanceInfo,
    ScoreInfo,
    FramesStepInfo,
)
from evaluation.common.metric_information.metric_info import (
    MetricInfo,
    SimpleMetricInfo,
)
from evaluation.repeatability.geometry.project_lines import make_projected_line_pairs
from evaluation.common.utils import filter_lines_by_score
from src.metrics.detection.vectorized import (
    repeatability_localization_error,
)
from src.metrics.detection.vectorized.distance.distance_factory import DistanceName
from src.typing import ArrayNx4, ArrayN, Array4x4


class UnscoredEvaluator(Evaluator):
    def __init__(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        depth_maps_paths: List[Path],
        euclidean_transforms: List[Array4x4[float]],
        calibration_matrix: Array4x4[float],
        frames_steps: List[int],
        distance_thresholds: List[float],
    ):
        self.pred_lines_batch = pred_lines_batch
        self.depth_maps_paths = depth_maps_paths
        self.euclidean_transforms = euclidean_transforms
        self.calibration_matrix = calibration_matrix
        self.frames_steps = frames_steps
        self.distance_thresholds = distance_thresholds

    def evaluate(self) -> List[MetricInfo]:
        results = []
        for dist in DistanceName:
            dist_name = dist.name
            for dist_threshold in self.distance_thresholds:
                for frames_step in self.frames_steps:
                    (
                        first_lines_batch,
                        second_lines_batch,
                        first_lines_projections_batch,
                        second_lines_projections_batch,
                    ) = make_projected_line_pairs(
                        self.pred_lines_batch,
                        self.euclidean_transforms,
                        self.calibration_matrix,
                        frames_step,
                        self.depth_maps_paths,
                    )

                    (
                        repeatability,
                        localization_error,
                    ) = repeatability_localization_error(
                        first_lines_batch,
                        second_lines_batch,
                        first_lines_projections_batch,
                        second_lines_projections_batch,
                        distance=dist_name,
                        distance_threshold=dist_threshold,
                    )

                    dist_info = DistanceInfo(
                        distance=dist_name, distance_threshold=dist_threshold
                    )
                    frame_step_info = FramesStepInfo(step=frames_step)

                    results.extend(
                        [
                            SimpleMetricInfo(
                                name="repeatability",
                                value=repeatability,
                                additional_information=[dist_info, frame_step_info],
                            ),
                            SimpleMetricInfo(
                                name="localization_error",
                                value=localization_error,
                                additional_information=[dist_info, frame_step_info],
                            ),
                        ]
                    )
        return results


class ScoredEvaluator(Evaluator):
    def __init__(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        depth_maps_paths: List[Path],
        euclidean_transforms: List[Array4x4[float]],
        calibration_matrix: Array4x4[float],
        frames_steps: List[int],
        scores_batch: List[ArrayN[float]],
        score_thresholds: List[float],
        distance_thresholds: List[float],
    ):
        self.pred_lines_batch = pred_lines_batch
        self.depth_maps_paths = depth_maps_paths
        self.euclidean_transforms = euclidean_transforms
        self.calibration_matrix = calibration_matrix
        self.frames_steps = frames_steps
        self.scores_batch = scores_batch
        self.score_thresholds = score_thresholds
        self.distance_thresholds = distance_thresholds

    def evaluate(self) -> List[MetricInfo]:
        results = []
        for score_threshold in self.score_thresholds:
            filtered_lines_batch = filter_lines_by_score(
                self.pred_lines_batch, self.scores_batch, score_threshold
            )

            metrics_info = UnscoredEvaluator(
                filtered_lines_batch,
                self.depth_maps_paths,
                self.euclidean_transforms,
                self.calibration_matrix,
                self.frames_steps,
                self.distance_thresholds,
            ).evaluate()

            for info in metrics_info:
                info.additional_information.append(
                    ScoreInfo(score_threshold=score_threshold)
                )
            results.extend(metrics_info)

        return results
