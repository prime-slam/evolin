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

from typing import List, Tuple, Union

from evolin.metrics.detection.vectorized.constants import (
    DISTANCE_NAMES,
    EVALUATION_RESOLUTION,
)
from evolin.metrics.detection.vectorized.distance.distance import Distance
from evolin.metrics.detection.vectorized.distance.distance_factory import (
    DistanceFactory,
)
from evolin.metrics.detection.vectorized.utils import docstring_arg
from evolin.typing import ArrayNx4
from evolin.metrics.detection.vectorized.tp_indicator import (
    VectorizedTPIndicator,
)

__all__ = [
    "vectorized_precision_recall_fscore",
    "vectorized_precision",
    "vectorized_recall",
    "vectorized_fscore",
]


class PrecisionRecall:
    """
    Class that calculates precision and recall
    over batches of predicted and ground truth lines
    """

    def __init__(
        self,
        tp_indicator: VectorizedTPIndicator,
    ):
        """
        Parameters
        ----------
        tp_indicator
            VectorizedTPIndicator object that indicates
            whether line is true positive or not
        """
        self.tp_indicator = tp_indicator

    def calculate(
        self,
        pred_lines_batch: List[ArrayNx4[float]],
        gt_lines_batch: List[ArrayNx4[float]],
    ) -> Tuple[float, float]:
        """
        Calculates vectorized precision and recall.

        Parameters
        ----------
        pred_lines_batch
            list of predicted lines for each image
        gt_lines_batch
            list of ground truth lines for each image

        Returns
        -------
        values
            precision and recall
        """

        tp = sum(
            self.tp_indicator.indicate(
                pred_lines, gt_lines, sort_predictions_by_distance=True
            ).sum()
            for pred_lines, gt_lines in zip(pred_lines_batch, gt_lines_batch)
        )
        gt_size = sum(len(gt_lines) for gt_lines in gt_lines_batch)
        pred_size = sum(len(pred_lines) for pred_lines in pred_lines_batch)
        fp = pred_size - tp

        recall = tp / gt_size
        precision = tp / (tp + fp) if tp + fp != 0 else 0

        return precision, recall


@docstring_arg(DISTANCE_NAMES, EVALUATION_RESOLUTION)
def vectorized_precision_recall_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> Tuple[float, float, float]:
    """
    Calculates vectorized precision, recall, and F-score.

    Parameters
    ----------
    pred_lines_batch
        list of predicted lines for each image
    gt_lines_batch
        list of ground truth lines for each image
    distance
        object of distance or distance name used
        to determine true positives ({0})
    distance_threshold
        threshold in pixels within which
        the line is considered to be true positive

    Returns
    -------
    values
        vectorized precision, recall, and F-score

    Notes
    -----
    Vectorized classification metrics are based on the vector representation of a line,
    that is, its representation as a pair of endpoints.
    Distance functions are used to determine if a line is True Positive.
    Further information can be found in papers [1]_ and [2]_.
    Each line should be represented as [x1, y1, x2, y2].
    Also, all lines must be scaled to the {1}x{1} resolution
    to eliminate the resolution factor affecting the distance threshold.

    Examples
    --------
    >>> import numpy as np
    >>> pred_lines_batch = [np.array([[1, 0, 2, 0], [2, 0, 1, 0], [5, 9, 11, 13]])]
    >>> gt_lines_batch = [np.array([[1, 0, 2, 0], [5, 10, 11, 12]])]
    >>> distance = "orthogonal"
    >>> distance_threshold = 5
    >>> precision, recall, fscore = vectorized_precision_recall_fscore(
    >>>     pred_lines_batch,
    >>>     gt_lines_batch,
    >>>     distance,
    >>>     distance_threshold
    >>> )

    References
    ----------
    .. [1] Zhou, Yichao, Haozhi Qi, and Yi Ma. "End-to-end wireframe parsing."
           Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
    .. [2] Pautrat, Rémi, et al. "SOLD2: Self-supervised occlusion-aware line description and detection."
           Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
    """

    distance = (
        DistanceFactory().from_string(distance)
        if isinstance(distance, str)
        else distance
    )

    tp_indicator = VectorizedTPIndicator(distance, distance_threshold)

    precision, recall = PrecisionRecall(tp_indicator=tp_indicator).calculate(
        pred_lines_batch,
        gt_lines_batch,
    )
    fscore = (
        2 * precision * recall / (precision + recall) if precision * recall != 0 else 0
    )

    return precision, recall, fscore


@docstring_arg(DISTANCE_NAMES, EVALUATION_RESOLUTION)
def vectorized_precision(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> float:
    """
    Calculates vectorized precision.

    Parameters
    ----------
    pred_lines_batch
        list of predicted lines for each image
    gt_lines_batch
        list of ground truth lines for each image
    distance
        object of distance or distance name used
        to determine true positives ({0})
    distance_threshold
        threshold in pixels within which
        the line is considered to be true positive

    Returns
    -------
    value
        vectorized precision

    Notes
    -----
    Vectorized classification metrics are based on the vector representation of a line,
    that is, its representation as a pair of endpoints.
    Distance functions are used to determine if a line is True Positive.
    Further information can be found in papers [1]_ and [2]_.
    Each line should be represented as [x1, y1, x2, y2].
    Also, all lines must be scaled to the {1}x{1} resolution
    to eliminate the resolution factor affecting the distance threshold.

    Examples
    --------
    >>> import numpy as np
    >>> pred_lines_batch = [np.array([[1, 0, 2, 0], [2, 0, 1, 0], [5, 9, 11, 13]])]
    >>> gt_lines_batch = [np.array([[1, 0, 2, 0], [5, 10, 11, 12]])]
    >>> distance = "orthogonal"
    >>> distance_threshold = 5
    >>> precision = vectorized_precision(
    >>>     pred_lines_batch,
    >>>     gt_lines_batch,
    >>>     distance,
    >>>     distance_threshold
    >>> )

    References
    ----------
    .. [1] Zhou, Yichao, Haozhi Qi, and Yi Ma. "End-to-end wireframe parsing."
           Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
    .. [2] Pautrat, Rémi, et al. "SOLD2: Self-supervised occlusion-aware line description and detection."
           Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
    """

    precision, _, _ = vectorized_precision_recall_fscore(
        pred_lines_batch, gt_lines_batch, distance, distance_threshold
    )

    return precision


@docstring_arg(DISTANCE_NAMES, EVALUATION_RESOLUTION)
def vectorized_recall(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> float:
    """
    Calculates vectorized recall.

    Parameters
    ----------
    pred_lines_batch
        list of predicted lines for each image
    gt_lines_batch
        list of ground truth lines for each image
    distance
        object of distance or distance name used
        to determine true positives ({0})
    distance_threshold
        threshold in pixels within which
        the line is considered to be true positive

    Returns
    -------
    value
        vectorized recall

    Notes
    -----
    Vectorized classification metrics are based on the vector representation of a line,
    that is, its representation as a pair of endpoints.
    Distance functions are used to determine if a line is True Positive.
    Further information can be found in papers [1]_ and [2]_.
    Each line should be represented as [x1, y1, x2, y2].
    Also, all lines must be scaled to the {1}x{1} resolution
    to eliminate the resolution factor affecting the distance threshold.

    Examples
    --------
    >>> import numpy as np
    >>> pred_lines_batch = [np.array([[1, 0, 2, 0], [2, 0, 1, 0], [5, 9, 11, 13]])]
    >>> gt_lines_batch = [np.array([[1, 0, 2, 0], [5, 10, 11, 12]])]
    >>> distance = "orthogonal"
    >>> distance_threshold = 5
    >>> recall = vectorized_recall(
    >>>     pred_lines_batch,
    >>>     gt_lines_batch,
    >>>     distance,
    >>>     distance_threshold
    >>> )

    References
    ----------
    .. [1] Zhou, Yichao, Haozhi Qi, and Yi Ma. "End-to-end wireframe parsing."
           Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
    .. [2] Pautrat, Rémi, et al. "SOLD2: Self-supervised occlusion-aware line description and detection."
           Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
    """

    _, recall, _ = vectorized_precision_recall_fscore(
        pred_lines_batch, gt_lines_batch, distance, distance_threshold
    )

    return recall


@docstring_arg(DISTANCE_NAMES, EVALUATION_RESOLUTION)
def vectorized_fscore(
    pred_lines_batch: List[ArrayNx4[float]],
    gt_lines_batch: List[ArrayNx4[float]],
    distance: Union[str, Distance] = "orthogonal",
    distance_threshold: float = 5,
) -> float:
    """
    Calculates vectorized F-score.

    Parameters
    ----------
    pred_lines_batch
        list of predicted lines for each image
    gt_lines_batch
        list of ground truth lines for each image
    distance
        object of distance or distance name used
        to determine true positives ({0})
    distance_threshold
        threshold in pixels within which
        the line is considered to be true positive

    Returns
    -------
    value
        vectorized F-score

    Notes
    -----
    Vectorized classification metrics are based on the vector representation of a line,
    that is, its representation as a pair of endpoints.
    Distance functions are used to determine if a line is True Positive.
    Further information can be found in papers [1]_ and [2]_.
    Each line should be represented as [x1, y1, x2, y2].
    Also, all lines must be scaled to the {1}x{1} resolution
    to eliminate the resolution factor affecting the distance threshold.

    Examples
    --------
    >>> import numpy as np
    >>> pred_lines_batch = [np.array([[1, 0, 2, 0], [2, 0, 1, 0], [5, 9, 11, 13]])]
    >>> gt_lines_batch = [np.array([[1, 0, 2, 0], [5, 10, 11, 12]])]
    >>> distance = "orthogonal"
    >>> distance_threshold = 5
    >>> fscore = vectorized_fscore(
    >>>     pred_lines_batch,
    >>>     gt_lines_batch,
    >>>     distance,
    >>>     distance_threshold
    >>> )

    References
    ----------
    .. [1] Zhou, Yichao, Haozhi Qi, and Yi Ma. "End-to-end wireframe parsing."
           Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
    .. [2] Pautrat, Rémi, et al. "SOLD2: Self-supervised occlusion-aware line description and detection."
           Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
    """
    _, _, fscore = vectorized_precision_recall_fscore(
        pred_lines_batch,
        gt_lines_batch,
        distance,
        distance_threshold,
    )

    return fscore
