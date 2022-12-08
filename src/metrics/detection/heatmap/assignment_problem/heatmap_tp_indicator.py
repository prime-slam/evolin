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

from src.metrics.detection.heatmap.assignment_problem.cost_matrix_creator import (
    CostMatrixCreator,
)
from src.typing import ArrayNxM


class HeatmapTPIndicator:
    def __init__(self, max_dist_diagonal_ratio: float = 0.01):
        self.max_dist_diag_ratio = max_dist_diagonal_ratio

    def indicate(
        self, gt_map: ArrayNxM[np.bool], pred_map: ArrayNxM[np.bool]
    ) -> ArrayNxM[np.bool]:
        if gt_map.shape != pred_map.shape:
            raise ValueError(
                "The gt heatmap and the predicted heatmap must have the same shape"
            )
        map_shape = gt_map.shape
        pred_map = pred_map
        diagonal = np.sqrt(sum(map(lambda x: x**2, map_shape)))
        outlier_cost = diagonal
        cost_matrix_creator = CostMatrixCreator(outlier_cost)
        max_dist = self.max_dist_diag_ratio * diagonal

        radius = np.ceil(max_dist).astype(int)
        gt_nodes_size = 0
        pred_nodes_size = 0

        pred_matchable = np.zeros(map_shape, dtype=bool)
        gt_node_to_pix = []
        gt_pix_to_node = np.full(map_shape, -1)
        pred_pix_to_node = np.full(map_shape, -1)
        edges = []
        epsilon = 1e-5

        for y1, x1 in zip(*gt_map.nonzero()):
            radius_mask = HeatmapTPIndicator.create_radius_mask(
                *map_shape, y1, x1, radius
            )
            pred_matchable_y, pred_matchable_x = (pred_map & radius_mask).nonzero()
            if pred_matchable_y.size != 0:
                gt_pix_to_node[y1, x1] = gt_nodes_size
                gt_node_to_pix.append((y1, x1))
                gt_nodes_size += 1
            for y2, x2 in zip(pred_matchable_y, pred_matchable_x):
                if not pred_matchable[y2, x2]:
                    pred_matchable[y2, x2] = True
                    pred_pix_to_node[y2, x2] = pred_nodes_size
                    pred_nodes_size += 1

                dist = (y1 - y2) ** 2 + (x1 - x2) ** 2
                i = gt_pix_to_node[y1, x1]
                j = pred_pix_to_node[y2, x2]
                w = max(np.sqrt(dist), epsilon)
                edges.append([i, j, w])
        tp_indicators = np.zeros(map_shape, dtype=bool)

        if len(edges) == 0:
            return tp_indicators

        edges = np.array(edges)
        cost_matrix = cost_matrix_creator.create(edges, gt_nodes_size, pred_nodes_size)
        matched_i, matched_j = min_weight_full_bipartite_matching(
            csr_matrix(cost_matrix)
        )

        # compute tp indicator map
        tp_indicators = np.zeros(map_shape, dtype=bool)
        for i, j in zip(matched_i, matched_j):
            if i >= gt_nodes_size or j >= pred_nodes_size:
                continue
            pix = gt_node_to_pix[i]
            tp_indicators[gt_node_to_pix[i]] = gt_map[pix]

        return tp_indicators

    @staticmethod
    def create_radius_mask(height, width, center_y, center_x, radius):
        y, x = np.ogrid[-center_y : height - center_y, -center_x : width - center_x]
        return x * x + y * y <= radius * radius
