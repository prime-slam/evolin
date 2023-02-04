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

from src.metrics.association.classification import precision_recall_fscore


@pytest.mark.parametrize(
    "pred_associations_batch, "
    "gt_associations_batch, "
    "expected_precision, "
    "expected_recall, "
    "expected_fscore",
    [
        (
            [],
            [],
            0.0,
            0.0,
            0.0,
        ),
        (
            [np.array([[0, 0], [1, 1]])],
            [np.array([[0, 0], [1, 1]])],
            1.0,
            1.0,
            1.0,
        ),
        (
            [np.array([[0, 0], [1, 1], [2, 2]])],
            [np.array([[0, 0], [1, 1]])],
            0.6666,
            1.0,
            0.8,
        ),
        (
            [np.array([[0, 0], [1, 1], [2, 2]])],
            [np.array([[0, 0], [1, 1], [3, 3]])],
            0.6666,
            0.6666,
            0.6666,
        ),
    ],
)
def test_classification(
    pred_associations_batch,
    gt_associations_batch,
    expected_precision,
    expected_recall,
    expected_fscore,
):
    epsilon = 1e-3
    actual_precision, actual_recall, actual_fscore = precision_recall_fscore(
        pred_associations_batch, gt_associations_batch
    )
    assert np.abs(actual_precision - expected_precision) < epsilon
    assert np.abs(actual_recall - expected_recall) < epsilon
    assert np.abs(actual_fscore - expected_fscore) < epsilon
