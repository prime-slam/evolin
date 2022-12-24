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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import distance_matrix
from scipy.stats import rankdata

from src.metrics.detection.heatmap.assignment_problem.cost_matrix_creator import (
    CostMatrixCreator,
)
from src.typing import ArrayNxM


class HeatmapTPIndicator:
    def __init__(self, max_dist_diagonal_ratio: float = 0.01):
        self.max_dist_diag_ratio = max_dist_diagonal_ratio

    def indicate(
        self, gt_map: ArrayNxM[bool], pred_map: ArrayNxM[bool]
    ) -> ArrayNxM[bool]:
        if gt_map.shape != pred_map.shape:
            raise ValueError(
                "The gt heatmap and the predicted heatmap must have the same shape"
            )
        map_shape = gt_map.shape
        diagonal = np.sqrt(sum(map(lambda x: x**2, map_shape)))
        outlier_cost = diagonal
        cost_matrix_creator = CostMatrixCreator(outlier_cost)
        max_dist = np.ceil(self.max_dist_diag_ratio * diagonal).astype(int)
        epsilon = 1e-5

        gt_pixels = np.column_stack(gt_map.nonzero())
        pred_pixels = np.column_stack(pred_map.nonzero())
        pixel_distances = distance_matrix(gt_pixels, pred_pixels)

        gt_pixels_index, pred_pixels_index = (pixel_distances <= max_dist).nonzero()
        gt_nodes = rankdata(gt_pixels_index, method="dense") - 1
        pred_nodes = rankdata(pred_pixels_index, method="dense") - 1
        weights = np.maximum(
            pixel_distances[gt_pixels_index, pred_pixels_index], epsilon
        )

        tp_indicators = np.zeros(map_shape, dtype=bool)

        if len(weights) == 0:
            return tp_indicators

        edges = np.column_stack([gt_nodes, pred_nodes, weights])
        gt_nodes_size = len(np.unique(gt_nodes))
        pred_nodes_size = len(np.unique(pred_nodes))
        cost_matrix = cost_matrix_creator.create(edges, gt_nodes_size, pred_nodes_size)

        gt_matched_nodes, pred_matched_nodes = min_weight_full_bipartite_matching(
            csr_matrix(cost_matrix)
        )

        gt_node_to_pixel_index = np.unique(gt_pixels_index)

        # compute tp indicator map
        tp_indicators = np.zeros(map_shape, dtype=bool)

        inliers_mask = (gt_matched_nodes < gt_nodes_size) & (
            pred_matched_nodes < pred_nodes_size
        )
        inlier_gt_nodes = gt_matched_nodes[inliers_mask]
        inlier_pixels = gt_pixels[gt_node_to_pixel_index[inlier_gt_nodes]]
        y, x = inlier_pixels[:, 0], inlier_pixels[:, 1]
        tp_indicators[y, x] = True

        return tp_indicators
