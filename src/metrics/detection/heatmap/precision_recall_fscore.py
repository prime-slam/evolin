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
import os

from joblib import Parallel, delayed
from tqdm.contrib import tzip
from typing import List, Tuple

from src.metrics.detection.heatmap import heatmap_precision_recall_curve
from src.metrics.detection.heatmap.basic_metrics import BasicMetrics
from src.metrics.detection.heatmap.tp_indicator import (
    HeatmapTPIndicator,
)
from src.metrics.detection.heatmap.utils import equally_sized, rasterize
from src.typing import ArrayNx4, ArrayN

__all__ = [
    "heatmap_recall",
    "heatmap_precision",
    "heatmap_precision_recall_fscore",
    "heatmap_max_fscore",
    "heatmap_fscore",
]


class PrecisionRecall:
    """
    Class that calculates heatmap precision and recall
    over batches of predicted and ground truth lines
    """

    def __init__(self):
        self.indicator = HeatmapTPIndicator()

    def calculate(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
        heights_batch: ArrayN[int],
        widths_batch: ArrayN[int],
    ) -> Tuple[float, float]:
        """
        Calculates heatmap precision and recall.

        Parameters
        ----------
        pred_lines_batch
            list of predicted lines for each image
        gt_lines_batch
            list of ground truth lines for each image
        heights_batch
            array of heights of each image
        widths_batch
            array of widths of each image

        Returns
        -------
        values
            heatmap precision and recall

        Notes
        -----
        Each line should be represented as [x1, y1, x2, y2].
        """
        if not equally_sized(
            [
                pred_lines_batch,
                gt_lines_batch,
                heights_batch,
                widths_batch,
            ]
        ):
            raise ValueError("All batches must be the same size")

        def calculate_statistics(pred_lines, gt_lines, height, width):
            gt_map = rasterize(gt_lines, height, width)
            pred_map = rasterize(pred_lines, height, width)
            tp_indicators_map = self.indicator.indicate(gt_map, pred_map)
            tp = tp_indicators_map.nnz
            fp = pred_map.nnz - tp_indicators_map.nnz
            gt_size = gt_map.nnz
            return BasicMetrics(tp, fp, gt_size)

        stats = sum(
            Parallel(n_jobs=os.cpu_count())(
                delayed(calculate_statistics)(pred_lines, gt_lines, height, width)
                for pred_lines, gt_lines, height, width in tzip(
                    pred_lines_batch,
                    gt_lines_batch,
                    heights_batch,
                    widths_batch,
                )
            )
        )

        tp = stats.tp
        fp = stats.fp
        gt = stats.gt

        recall = tp / gt
        precision = tp / (tp + fp) if tp + fp != 0 else 0

        return precision, recall


def heatmap_precision_recall_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
) -> Tuple[float, float, float]:
    """
    Calculates heatmap precision, recall, and F-score.

    Parameters
    ----------
    pred_lines_batch
        list of predicted lines for each image
    gt_lines_batch
        list of ground truth lines for each image
    heights_batch
        array of heights of each image
    widths_batch
        array of widths of each image

    Returns
    -------
    values
        heatmap precision, recall, and F-score

    Notes
    -----
    Each line should be represented as [x1, y1, x2, y2].
    In the case of a raster representation of lines (or heatmap),
    it is possible to consider detection to classify each pixel
    from the point of view of belonging to any line.
    See [1]_, [2]_ for more information.

    Examples
    --------
    >>> import numpy as np
    >>> pred_lines_batch = [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])]
    >>> gt_lines_batch = [np.array([[5, 0, 0, 5]])]
    >>> heights_batch = np.array([5])
    >>> widths_batch = np.array([5])
    >>> precision, recall, fscore = heatmap_precision_recall_fscore(
    >>>     pred_lines_batch,
    >>>     gt_lines_batch,
    >>>     heights_batch,
    >>>     widths_batch,
    >>> )

    References
    ----------
    .. [1] Huang, Kun, et al. "Learning to parse wireframes in images of man-made environments."
           Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    .. [2] Martin, David R., Charless C. Fowlkes, and Jitendra Malik.
           "Learning to detect natural image boundaries using local brightness, color, and texture cues."
           IEEE transactions on pattern analysis and machine intelligence 26.5 (2004): 530-549.
    """

    precision, recall = PrecisionRecall().calculate(
        pred_lines_batch, gt_lines_batch, heights_batch, widths_batch
    )
    fscore = (
        2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    )

    return precision, recall, fscore


def heatmap_precision(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
) -> float:
    """
    Calculates heatmap precision.

    Parameters
    ----------
    pred_lines_batch
        list of predicted lines for each image
    gt_lines_batch
        list of ground truth lines for each image
    heights_batch
        array of heights of each image
    widths_batch
        array of widths of each image

    Returns
    -------
    values
        heatmap precision

    Notes
    -----
    Each line should be represented as [x1, y1, x2, y2].
    In the case of a raster representation of lines (or heatmap),
    it is possible to consider detection to classify each pixel
    from the point of view of belonging to any line.
    See [1]_, [2]_ for more information.

    Examples
    --------
    >>> import numpy as np
    >>> pred_lines_batch = [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])]
    >>> gt_lines_batch = [np.array([[5, 0, 0, 5]])]
    >>> heights_batch = np.array([5])
    >>> widths_batch = np.array([5])
    >>> precision = heatmap_precision(
    >>>     pred_lines_batch,
    >>>     gt_lines_batch,
    >>>     heights_batch,
    >>>     widths_batch,
    >>> )

    References
    ----------
    .. [1] Huang, Kun, et al. "Learning to parse wireframes in images of man-made environments."
           Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    .. [2] Martin, David R., Charless C. Fowlkes, and Jitendra Malik.
           "Learning to detect natural image boundaries using local brightness, color, and texture cues."
           IEEE transactions on pattern analysis and machine intelligence 26.5 (2004): 530-549.
    """

    precision, _ = PrecisionRecall().calculate(
        pred_lines_batch, gt_lines_batch, heights_batch, widths_batch
    )

    return precision


def heatmap_recall(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
) -> float:
    """
    Calculates heatmap recall.

    Parameters
    ----------
    pred_lines_batch
        list of predicted lines for each image
    gt_lines_batch
        list of ground truth lines for each image
    heights_batch
        array of heights of each image
    widths_batch
        array of widths of each image

    Returns
    -------
    values
        heatmap recall

    Notes
    -----
    Each line should be represented as [x1, y1, x2, y2].
    In the case of a raster representation of lines (or heatmap),
    it is possible to consider detection to classify each pixel
    from the point of view of belonging to any line.
    See [1]_, [2]_ for more information.

    Examples
    --------
    >>> import numpy as np
    >>> pred_lines_batch = [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])]
    >>> gt_lines_batch = [np.array([[5, 0, 0, 5]])]
    >>> heights_batch = np.array([5])
    >>> widths_batch = np.array([5])
    >>> recall = heatmap_recall(
    >>>     pred_lines_batch,
    >>>     gt_lines_batch,
    >>>     heights_batch,
    >>>     widths_batch,
    >>> )

    References
    ----------
    .. [1] Huang, Kun, et al. "Learning to parse wireframes in images of man-made environments."
           Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    .. [2] Martin, David R., Charless C. Fowlkes, and Jitendra Malik.
           "Learning to detect natural image boundaries using local brightness, color, and texture cues."
           IEEE transactions on pattern analysis and machine intelligence 26.5 (2004): 530-549.
    """
    _, recall = PrecisionRecall().calculate(
        pred_lines_batch, gt_lines_batch, heights_batch, widths_batch
    )

    return recall


def heatmap_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
):
    """
    Calculates heatmap F-score.

    Parameters
    ----------
    pred_lines_batch
        list of predicted lines for each image
    gt_lines_batch
        list of ground truth lines for each image
    heights_batch
        array of heights of each image
    widths_batch
        array of widths of each image

    Returns
    -------
    values
        heatmap F-score

    Notes
    -----
    Each line should be represented as [x1, y1, x2, y2].
    In the case of a raster representation of lines (or heatmap),
    it is possible to consider detection to classify each pixel
    from the point of view of belonging to any line.
    See [1]_, [2]_ for more information.

    Examples
    --------
    >>> import numpy as np
    >>> pred_lines_batch = [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])]
    >>> gt_lines_batch = [np.array([[5, 0, 0, 5]])]
    >>> heights_batch = np.array([5])
    >>> widths_batch = np.array([5])
    >>> fscore = heatmap_fscore(
    >>>     pred_lines_batch,
    >>>     gt_lines_batch,
    >>>     heights_batch,
    >>>     widths_batch,
    >>> )

    References
    ----------
    .. [1] Huang, Kun, et al. "Learning to parse wireframes in images of man-made environments."
           Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    .. [2] Martin, David R., Charless C. Fowlkes, and Jitendra Malik.
           "Learning to detect natural image boundaries using local brightness, color, and texture cues."
           IEEE transactions on pattern analysis and machine intelligence 26.5 (2004): 530-549.
    """
    _, _, fscore = heatmap_precision_recall_fscore(
        pred_lines_batch, gt_lines_batch, heights_batch, widths_batch
    )

    return fscore


def heatmap_max_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    line_scores_batch: List[ArrayNx4[float]],
    heights_batch: ArrayN[int],
    widths_batch: ArrayN[int],
    thresholds: ArrayN[int],
):
    """
    Calculates maximum F-score among all line score thresholds.

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
    Notes
    -----
    Each line should be represented as [x1, y1, x2, y2].
    In the case of a raster representation of lines (or heatmap),
    it is possible to consider detection to classify each pixel
    from the point of view of belonging to any line.
    See [1]_, [2]_ for more information.

    Returns
    -------
    value
        maximum F-score

    Examples
    --------
    >>> import numpy as np
    >>> pred_lines_batch = [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])]
    >>> gt_lines_batch = [np.array([[5, 0, 0, 5], [0, 0, 5, 5]])]
    >>> scores_batch = [np.array([0.1, 1])]
    >>> heights_batch = np.array([5])
    >>> widths_batch = np.array([5])
    >>> thresholds = np.array([0.0, 0.2])
    >>> aph = heatmap_max_fscore(
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

    fscore = np.zeros(precision.size, dtype=float)
    nonzero_mask = precision + recall != 0
    fscore[nonzero_mask] = (
        2
        * precision[nonzero_mask]
        * recall[nonzero_mask]
        / (precision[nonzero_mask] + recall[nonzero_mask])
    )

    return np.max(fscore)
