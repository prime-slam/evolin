# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
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

from pathlib import Path
from skimage.io import imread
from typing import List

from evaluation.common.evaluator_base import Evaluator
from evaluation.common.geometry.project_lines import get_3d_lines
from evaluation.common.geometry.relative_pose import RelativePoseEstimator
from evaluation.common.metric_information.additional_info import (
    FramesStepInfo,
    ThresholdInfo,
)
from evaluation.common.metric_information.metric_info import (
    MetricInfo,
    SimpleMetricInfo,
)
from evaluation.common.utils import clip_lines
from src.metrics.association.pose_error import (
    pose_error,
    angular_pose_error_auc,
)
from src.typing import ArrayNx4, Array4x4


class PoseErrorEvaluator(Evaluator):
    def __init__(
        self,
        lines_batch: List[ArrayNx4[float]],
        associations_batch: List[ArrayNx4[float]],
        depth_maps_paths: List[Path],
        gt_absolute_poses: List[Array4x4[float]],
        calibration_matrix: Array4x4[float],
        pose_error_auc_thresholds: List[float],
        depth_scaler: float = 5000,
    ):
        self.lines_batch = lines_batch
        self.associations_batch = associations_batch
        self.images_number = len(depth_maps_paths)
        self.frames_step = self.images_number - len(associations_batch) + 1
        self.frames_pairs = list(
            zip(range(self.images_number), range(self.frames_step, self.images_number))
        )
        self.depth_maps_paths = depth_maps_paths
        self.gt_relative_poses = [
            gt_absolute_poses[second] @ np.linalg.inv(gt_absolute_poses[first])
            for first, second in self.frames_pairs
        ]
        self.calibration_matrix = calibration_matrix
        self.depth_scaler = depth_scaler
        self.pose_error_auc_thresholds = pose_error_auc_thresholds

    def evaluate(self) -> List[MetricInfo]:
        angular_rotation_errors = []
        angular_translation_errors = []
        absolute_translation_errors = []
        est_relative_poses = self.__estimate_relative_poses()

        for gt_pose, est_pose in zip(self.gt_relative_poses, est_relative_poses):
            if est_pose is not None:
                (
                    angular_translation_error_,
                    angular_rotation_error_,
                    absolute_translation_error_,
                ) = pose_error(gt_pose, est_pose)
                if not np.isnan(angular_translation_error_) and not np.isnan(
                    angular_rotation_error_
                ):
                    angular_rotation_errors.append(angular_rotation_error_)
                    angular_translation_errors.append(angular_translation_error_)
                if not np.isnan(absolute_translation_error_):
                    absolute_translation_errors.append(absolute_translation_error_)

        angular_rotation_errors = np.array(angular_rotation_errors)
        angular_translation_errors = np.array(angular_translation_errors)
        absolute_translation_errors = np.array(absolute_translation_errors)

        frame_step_info = FramesStepInfo(step=self.frames_step)

        results = [
            SimpleMetricInfo(
                name="median_angular_rotation_error",
                value=np.median(angular_rotation_errors),
                additional_information=[frame_step_info],
            ),
            SimpleMetricInfo(
                name="median_angular_translation_error",
                value=np.median(angular_translation_errors),
                additional_information=[frame_step_info],
            ),
            SimpleMetricInfo(
                name="median_absolute_translation_error",
                value=np.median(absolute_translation_errors),
                additional_information=[frame_step_info],
            ),
        ]

        for threshold in self.pose_error_auc_thresholds:
            threshold_info = ThresholdInfo(threshold=threshold)
            auc = angular_pose_error_auc(
                angular_rotation_errors, angular_translation_errors, threshold
            )
            results.append(
                SimpleMetricInfo(
                    name="pose_error_auc",
                    value=auc,
                    additional_information=[threshold_info, frame_step_info],
                )
            )

        return results

    def __estimate_relative_poses(self):
        est_relative_poses = []
        for associations, frames_pair in zip(
            self.associations_batch, self.frames_pairs
        ):
            first_frame, second_frame = frames_pair
            pose = None

            if associations.size > 2:
                first_index = associations[:, 0]
                second_index = associations[:, 1]
                first_depth = (
                    imread(self.depth_maps_paths[first_frame]) / self.depth_scaler
                )
                second_depth = (
                    imread(self.depth_maps_paths[second_frame]) / self.depth_scaler
                )
                first_height, first_width = first_depth.shape[:2]
                second_height, second_width = first_depth.shape[:2]
                first_lines = (
                    clip_lines(
                        self.lines_batch[first_frame],
                        width=first_width,
                        height=first_height,
                    )
                    .astype(int)
                    .reshape((-1, 2, 2))[first_index]
                )
                second_lines = (
                    clip_lines(
                        self.lines_batch[second_frame],
                        width=second_width,
                        height=second_height,
                    )
                    .astype(int)
                    .reshape((-1, 2, 2))[second_index]
                )
                lines_3d_2 = get_3d_lines(
                    second_lines, second_depth, self.calibration_matrix
                )
                pose = RelativePoseEstimator().estimate(
                    first_lines, second_lines, lines_3d_2, self.calibration_matrix
                )
            est_relative_poses.append(pose)

        return est_relative_poses
