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

from evolin.metrics.detection.heatmap import heatmap_fscore, heatmap_max_fscore


@pytest.mark.parametrize(
    "pred_lines_batch, "
    "gt_lines_batch, "
    "heights_batch, "
    "widths_batch, "
    "expected_aph",
    [
        (
            [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])],
            [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])],
            np.array([5]),
            np.array([5]),
            1.0,
        ),
        (
            [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])],
            [np.array([[5, 0, 0, 5]])],
            np.array([5]),
            np.array([5]),
            0.7142,
        ),
        (
            [np.array([])],
            [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])],
            np.array([5]),
            np.array([5]),
            0.0,
        ),
    ],
)
def test_heatmap_fscore(
    pred_lines_batch,
    gt_lines_batch,
    heights_batch,
    widths_batch,
    expected_aph,
):
    epsilon = 1e-3
    actual_aph = heatmap_fscore(
        pred_lines_batch,
        gt_lines_batch,
        heights_batch,
        widths_batch,
    )
    assert np.abs(actual_aph - expected_aph) < epsilon


@pytest.mark.parametrize(
    "pred_lines_batch, "
    "gt_lines_batch, "
    "scores_batch, "
    "heights_batch, "
    "widths_batch, "
    "thresholds, "
    "expected_fscore",
    [
        (
            [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])],
            [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])],
            [np.array([0.1, 1])],
            np.array([5]),
            np.array([5]),
            np.array([0.0, 0.2]),
            1.0,
        )
    ],
)
def test_heatmap_max_fscore(
    pred_lines_batch,
    gt_lines_batch,
    scores_batch,
    heights_batch,
    widths_batch,
    thresholds,
    expected_fscore,
):
    epsilon = 1e-5
    actual_fscore = heatmap_max_fscore(
        pred_lines_batch,
        gt_lines_batch,
        scores_batch,
        heights_batch,
        widths_batch,
        thresholds,
    )
    assert np.abs(actual_fscore - expected_fscore) < epsilon
