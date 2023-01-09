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
import pytest

from src.metrics.detection.vectorized import EVALUATION_RESOLUTION
from src.metrics.detection.vectorized import repeatability_localization_error


def test_repeatability_localization_error():
    width = 640
    height = 320

    # calibration matrix
    K = np.array(
        [[1, 0, width / 2, 0], [0, 1, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    # transforms from world to camera frames
    E1 = np.diag(np.full(4, 1))
    E2 = np.array(
        [
            [-0.999762, 0.000000, -0.021799, 0.790932],
            [0.000000, 1.000000, 0.000000, 1.300000],
            [0.021799, 0.000000, -0.999762, 1.462270],
            [0, 0, 0, 1],
        ]
    )

    # camera matrices
    P1 = K @ E1
    P2 = K @ E2

    # endpoints
    start_point1_3D = [5, 2, 1, 1]
    end_point1_3D = [25, 35, 1, 1]
    start_point2_3D = [7, 75, 1, 1]
    end_point2_3D = [25, 35, 1, 1]

    start_point1_cam1 = P1 @ start_point1_3D
    end_point1_cam1 = P1 @ end_point1_3D
    start_point2_cam1 = P1 @ start_point2_3D
    end_point2_cam1 = P1 @ end_point2_3D

    start_point1_cam2 = P2 @ start_point1_3D
    end_point1_cam2 = P2 @ end_point1_3D
    start_point2_cam2 = P2 @ start_point2_3D
    end_point2_cam2 = P2 @ end_point2_3D

    start_point1_cam2 /= start_point1_cam2[2]
    end_point1_cam2 /= end_point1_cam2[2]
    start_point2_cam2 /= end_point1_cam2[2]
    end_point2_cam2 /= end_point2_cam2[2]

    lines1 = np.array(
        [
            [start_point1_cam1[:2], end_point1_cam1[:2]],
            [start_point2_cam1[:2], end_point2_cam1[:2]],
        ]
    )
    lines2 = np.array(
        [
            [start_point1_cam2[:2], end_point1_cam2[:2]],
            [start_point2_cam2[:2], end_point2_cam2[:2]],
        ]
    )
    lines1 = lines1.reshape(-1, 4)
    lines2 = lines2.reshape(-1, 4)

    x_index = [0, 2]
    y_index = [1, 3]
    lines1[..., x_index] *= EVALUATION_RESOLUTION / width
    lines2[..., x_index] *= EVALUATION_RESOLUTION / width
    lines1[..., y_index] *= EVALUATION_RESOLUTION / height
    lines2[..., y_index] *= EVALUATION_RESOLUTION / height

    expected_repeatability = 1.0
    expected_localization_error = 0.0

    # perfect reprojection
    actual_repeatability, actual_localization_error = repeatability_localization_error(
        first_lines_batch=[lines1],
        second_lines_batch=[lines2],
        first_lines_projections_batch=[lines2],
        second_lines_projections_batch=[lines1],
    )

    assert actual_repeatability == expected_repeatability
    assert actual_localization_error == expected_localization_error
