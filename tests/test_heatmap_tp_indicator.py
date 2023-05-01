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

from scipy.sparse import dok_matrix

from evolin.metrics.detection.heatmap.tp_indicator import (
    HeatmapTPIndicator,
)


@pytest.mark.parametrize(
    "gt_map, pred_map, expected_tp_map",
    [
        (
            dok_matrix(np.diag(np.full(5, True))),
            dok_matrix(np.diag(np.full(5, True))),
            dok_matrix(np.diag(np.full(5, True))),
        ),
        (
            dok_matrix(np.zeros((5, 5))),
            dok_matrix(np.diag(np.full(5, True))),
            dok_matrix(np.zeros((5, 5))),
        ),
        (
            dok_matrix(np.ones((5, 5))),
            dok_matrix(np.diag(np.full(5, True))),
            dok_matrix(np.diag(np.full(5, True))),
        ),
    ],
)
def test_heatmap_tp_indicator(gt_map, pred_map, expected_tp_map):
    actual_tp_map = HeatmapTPIndicator().indicate(gt_map, pred_map)
    assert (actual_tp_map.toarray() == expected_tp_map.toarray()).all()
