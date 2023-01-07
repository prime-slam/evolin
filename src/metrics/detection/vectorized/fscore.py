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

from typing import List, Union

from src.metrics.detection.vectorized import DISTANCE_NAMES
from src.metrics.detection.vectorized.distance.distance import Distance
from src.metrics.detection.vectorized.precision_recall import (
    vectorized_precision_recall,
)
from src.metrics.detection.vectorized.utils import docstring_arg
from src.typing import ArrayNx4

__all__ = ["vectorized_fscore"]


@docstring_arg(DISTANCE_NAMES)
def vectorized_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> float:
    """
    Calculates vectorized F-Score
    :param pred_lines_batch: list of predicted lines for each image
    :param gt_lines_batch: list of ground truth lines for each image
    :param distance: distance object or distance name used
    to determine true positives ({0})
    :param distance_threshold: threshold in pixels within which
    the line is considered to be true positive
    :return: vectorized F-Score value
    """
    precision, recall = vectorized_precision_recall(
        pred_lines_batch,
        gt_lines_batch,
        distance,
        distance_threshold,
    )

    fscore = (
        2 * precision * recall / (precision + recall) if precision * recall != 0 else 0
    )

    return fscore
