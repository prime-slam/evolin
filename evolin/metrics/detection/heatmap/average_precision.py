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

from typing import List

from evolin.metrics.detection.heatmap.precision_recall_curve import (
    heatmap_precision_recall_curve,
)
from evolin.typing import ArrayNx4, ArrayN

__all__ = ["heatmap_average_precision"]


def heatmap_average_precision(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    line_scores_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
    thresholds: ArrayN[float],
) -> float:
    """
    Calculates Heatmap Average Precision (:math:`AP^H`)

    Parameters
    ----------
    pred_lines_batch
        list of predicted lines for each image
    gt_lines_batch
        list of ground truth lines for each image
    line_scores_batch
        list of predicted lines scores for each image
    heights_batch
        array of heights of each image
    widths_batch
        array of widths of each image
    thresholds
        array of line scores thresholds to filter predicted lines

    Notes
    -----
    Initially, each line should be represented as [x1, y1, x2, y2].
    In the case of a raster representation of lines (or heatmap),
    it is possible to consider detection to classify each pixel
    from the point of view of belonging to any line.
    See [1]_, [2]_ for more information.

    Returns
    -------
    value
        Heatmap Average Precision

    Examples
    --------
    >>> import numpy as np
    >>> pred_lines_batch = [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])]
    >>> gt_lines_batch = [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])]
    >>> scores_batch = [np.array([0.1, 1])]
    >>> heights_batch = np.array([5])
    >>> widths_batch = np.array([5])
    >>> thresholds = np.array([0.0, 0.2])
    >>> aph = heatmap_average_precision(
    >>>     pred_lines_batch,
    >>>     gt_lines_batch,
    >>>     scores_batch,
    >>>     heights_batch,
    >>>     widths_batch,
    >>>     thresholds,
    >>> )

    References
    ----------
    .. [1] Huang, Kun, et al. "Learning to parse wireframes in images of man-made environments."
           Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    .. [2] Martin, David R., Charless C. Fowlkes, and Jitendra Malik.
           "Learning to detect natural image boundaries using local brightness, color, and texture cues."
           IEEE transactions on pattern analysis and machine intelligence 26.5 (2004): 530-549.
    """

    precision, recall = heatmap_precision_recall_curve(
        pred_lines_batch,
        gt_lines_batch,
        line_scores_batch,
        heights_batch,
        widths_batch,
        thresholds,
    )

    # AP is the area under the PR Curve
    return np.trapz(x=recall, y=precision)
