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
from typing import List, Tuple

import numpy as np
from skimage import io

from evaluation.common.utils import clip_lines, is_nonzero_length, scale_lines
from evolin.metrics.detection.vectorized import EVALUATION_RESOLUTION
from evolin.typing import (
    ArrayNx4,
    Array4x4,
    ArrayNxM,
    ArrayNx2,
    ArrayNxN,
    ArrayNx2x2,
    ArrayNx3,
    ArrayNx6,
)


def get_3d_points(
    points: ArrayNx2[float],
    depth_map: ArrayNxM[float],
    calibration_matrix: Array4x4[float],
) -> ArrayNx3[float]:
    fx = calibration_matrix[0, 0]
    fy = calibration_matrix[1, 1]
    cx = calibration_matrix[0, 2]
    cy = calibration_matrix[1, 2]
    x, y = points[:, 0], points[:, 1]
    depths = depth_map[y, x]
    nonzero_depths = depths != 0
    z_3d = depths
    x_3d = np.zeros(len(z_3d))
    y_3d = np.zeros(len(z_3d))
    x_3d[nonzero_depths] = (x[nonzero_depths] - cx) / fx * z_3d[nonzero_depths]
    y_3d[nonzero_depths] = (y[nonzero_depths] - cy) / fy * z_3d[nonzero_depths]
    # set np.inf if depth is zero
    z_3d[~nonzero_depths] = np.inf
    x_3d[~nonzero_depths] = np.inf
    y_3d[~nonzero_depths] = np.inf

    return np.column_stack([x_3d, y_3d, z_3d])


def get_3d_lines(
    lines: ArrayNx2x2[float],
    depth_map: ArrayNxM[float],
    calibration_matrix: Array4x4[float],
) -> ArrayNx6[float]:
    start_points = lines[:, 0]
    end_points = lines[:, 1]
    start_points_3d = get_3d_points(start_points, depth_map, calibration_matrix)
    end_points_3d = get_3d_points(end_points, depth_map, calibration_matrix)
    lines_3d = np.column_stack([start_points_3d, end_points_3d])

    return lines_3d


def project_points(
    points: ArrayNx2[float], projection_matrix: Array4x4[float], depths: ArrayNx2
) -> ArrayNx2[float]:
    ones_column = np.ones((len(points), 1))
    points_homo = np.concatenate(
        (points, ones_column, 1 / depths.reshape(-1, 1)), axis=1
    )
    projected_points_homo = projection_matrix @ points_homo.T
    projected_points_homo /= projected_points_homo[2]
    projected_points = projected_points_homo.T[..., :2]

    return projected_points


def project_lines(
    lines: ArrayNx4[float],
    projection_matrix: Array4x4[float],
    depth_map: ArrayNxM[float],
) -> ArrayNx4[float]:
    start_points = np.rint(lines[..., :2]).astype(int)
    end_points = np.rint(lines[..., 2:]).astype(int)

    start_points_depths = depth_map[start_points[..., 1], start_points[..., 0]]
    end_points_depths = depth_map[end_points[..., 1], end_points[..., 0]]

    nonzero_depth_map = (start_points_depths != 0) & (end_points_depths != 0)
    start_points = start_points[nonzero_depth_map]
    end_points = end_points[nonzero_depth_map]
    start_points_depths = start_points_depths[nonzero_depth_map]
    end_points_depths = end_points_depths[nonzero_depth_map]

    projected_start_points = project_points(
        start_points, projection_matrix, start_points_depths
    )
    projected_end_points = project_points(
        end_points, projection_matrix, end_points_depths
    )

    nonzero_length_mask = np.logical_and.reduce(
        projected_start_points != projected_end_points, axis=-1
    )

    projected_lines = np.concatenate(
        (
            projected_start_points[nonzero_length_mask],
            projected_end_points[nonzero_length_mask],
        ),
        axis=1,
    )

    return projected_lines


def make_projected_line_pairs(
    pred_lines_batch: List[ArrayNx4[float]],
    euclidean_transforms: List[ArrayNxN[float]],
    calibration_matrix: Array4x4[float],
    frames_step: int,
    depth_map_paths: List[Path],
    depth_scaler: float = 5000,
) -> Tuple[
    List[ArrayNx4[float]],
    List[ArrayNx4[float]],
    List[ArrayNx4[float]],
    List[ArrayNx4[float]],
]:
    num_frames = len(pred_lines_batch)

    first_lines_batch = []
    second_lines_batch = []
    first_lines_projections_batch = []
    second_lines_projections_batch = []
    for i in range(num_frames - frames_step):
        first_frame = i
        second_frame = i + frames_step

        first_lines = pred_lines_batch[first_frame].copy()
        second_lines = pred_lines_batch[second_frame].copy()
        first_depth_map = io.imread(depth_map_paths[first_frame]) / depth_scaler
        height1, width1 = first_depth_map.shape[:2]
        second_depth_map = io.imread(depth_map_paths[second_frame]) / depth_scaler
        height2, width2 = second_depth_map.shape[:2]

        if first_lines.size != 0:
            first_lines = clip_lines(first_lines, height1, width1)
        if second_lines.size != 0:
            second_lines = clip_lines(second_lines, height2, width2)

        E1 = euclidean_transforms[first_frame]
        E2 = euclidean_transforms[second_frame]

        P1 = calibration_matrix @ E1
        P2 = calibration_matrix @ E2

        M = P2 @ np.linalg.inv(P1)
        M_inv = P1 @ np.linalg.inv(P2)

        if first_lines.size != 0:
            first_lines_projections = clip_lines(
                project_lines(first_lines, M, first_depth_map), height1, width1
            )
            first_lines_projections = first_lines_projections[
                is_nonzero_length(first_lines_projections)
            ]

            first_lines_projections_batch.append(
                scale_lines(
                    first_lines_projections,
                    x_scaler=EVALUATION_RESOLUTION / width1,
                    y_scaler=EVALUATION_RESOLUTION / height1,
                )
            )

        if second_lines.size != 0:
            second_lines_projections = clip_lines(
                project_lines(second_lines, M_inv, second_depth_map), height2, width2
            )
            second_lines_projections = second_lines_projections[
                is_nonzero_length(second_lines_projections)
            ]
            second_lines_projections_batch.append(
                scale_lines(
                    second_lines_projections,
                    x_scaler=EVALUATION_RESOLUTION / width2,
                    y_scaler=EVALUATION_RESOLUTION / height2,
                )
            )

        first_lines_batch.append(
            scale_lines(
                first_lines,
                x_scaler=EVALUATION_RESOLUTION / width1,
                y_scaler=EVALUATION_RESOLUTION / height1,
            )
            if first_lines.size != 0
            else first_lines
        )
        second_lines_batch.append(
            scale_lines(
                second_lines,
                x_scaler=EVALUATION_RESOLUTION / width2,
                y_scaler=EVALUATION_RESOLUTION / height2,
            )
            if second_lines.size != 0
            else second_lines
        )

    return (
        first_lines_batch,
        second_lines_batch,
        first_lines_projections_batch,
        second_lines_projections_batch,
    )
