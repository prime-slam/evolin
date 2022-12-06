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

from typing import Tuple

from src.metrics.detection.vectorized.distance.distance import Distance
from src.typing import ArrayNx2x2, ArrayNxM, ArrayNx2, ArrayNxMx2


class OrthogonalDistance(Distance):
    def __init__(self, min_overlap: float = 0.5):
        self.min_overlap = min_overlap

    def calculate(
        self, first_lines: ArrayNx2x2[np.float], second_lines: ArrayNx2x2[np.float]
    ) -> ArrayNxM[np.float]:
        first_lines_number, second_lines_number = len(first_lines), len(second_lines)

        # Average Orthogonal Line Distance calculation
        first_endpoints = first_lines.reshape(-1, 2)
        second_endpoints = second_lines.reshape(-1, 2)

        (
            second_to_first_endpoint_distances,
            second_endpoints_projection_offsets,
        ) = OrthogonalDistance.__calculate_orthogonal_distance(
            first_lines, second_endpoints
        )

        (
            first_to_second_endpoint_distances,
            first_endpoints_projection_offsets,
        ) = OrthogonalDistance.__calculate_orthogonal_distance(
            second_lines, first_endpoints
        )

        second_to_first_line_distances = second_to_first_endpoint_distances.reshape(
            (first_lines_number, second_lines_number, 2)
        ).sum(axis=2)

        first_to_second_line_distances = first_to_second_endpoint_distances.reshape(
            (second_lines_number, first_lines_number, 2)
        ).sum(axis=2)

        orthogonal_distances = (
            second_to_first_line_distances + first_to_second_line_distances.T
        ) / 2

        # Average Overlapping Ratio Calculation
        second_endpoints_projection_offsets = (
            second_endpoints_projection_offsets.reshape(
                (first_lines_number, second_lines_number, 2)
            )
        )
        second_endpoints_overlaps = OrthogonalDistance.__calculate_endpoints_overlap(
            second_endpoints_projection_offsets
        )
        first_endpoints_projection_offsets = first_endpoints_projection_offsets.reshape(
            (second_lines_number, first_lines_number, 2)
        )
        first_endpoints_overlaps = OrthogonalDistance.__calculate_endpoints_overlap(
            first_endpoints_projection_offsets
        ).T
        overlaps = (second_endpoints_overlaps + first_endpoints_overlaps) / 2

        orthogonal_distances[overlaps < self.min_overlap] = np.inf

        return orthogonal_distances

    @staticmethod
    def __calculate_orthogonal_distance(
        lines: ArrayNx2x2[np.float], points: ArrayNx2[np.float]
    ) -> Tuple[ArrayNxM[np.float], ArrayNxM[np.float]]:
        # Using line parametrisation endpoint1 + offset (endpoint2 - endpoint1)

        direction = (lines[:, 1] - lines[:, 0])[:, np.newaxis]
        offset = ((points[np.newaxis] - lines[:, np.newaxis, 0]) * direction).sum(
            axis=2
        ) / np.linalg.norm(direction, axis=2) ** 2

        projection = lines[:, np.newaxis, 0] + offset[..., np.newaxis] * direction
        orthogonal_distance = np.linalg.norm(projection - points[np.newaxis], axis=2)

        return orthogonal_distance, offset

    @staticmethod
    def __calculate_endpoints_overlap(
        endpoints_offsets: ArrayNxMx2[np.float],
    ) -> ArrayNxM[np.float]:
        # Overlap of a projection of a line on another line calculation
        # Original segments have endpoints with offsets 0 and 1 in the line parametrization
        endpoints_offsets = np.sort(endpoints_offsets, axis=-1)
        overlap = (
            (endpoints_offsets[..., 1] > 0)
            * (endpoints_offsets[..., 0] < 1)
            * (
                np.minimum(endpoints_offsets[..., 1], 1)
                - np.maximum(endpoints_offsets[..., 0], 0)
            )
        )
        return overlap
