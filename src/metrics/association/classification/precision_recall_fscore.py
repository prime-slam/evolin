# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
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

from tqdm.contrib import tzip
from typing import List, Tuple

from src.metrics.association.classification.tp_indicator import TPIndicator
from src.metrics.detection.heatmap.utils import equally_sized
from src.typing import ArrayNx2

__all__ = ["precision_recall_fscore", "precision", "recall", "fscore"]


class PrecisionRecall:
    """
    Class that calculates heatmap precision and recall
    over batches of predicted and ground truth lines
    """

    def __init__(self):
        self.indicator = TPIndicator()

    def calculate(
        self,
        pred_associations_batch: List[ArrayNx2[float]],
        gt_associations_batch: List[ArrayNx2[float]],
    ) -> Tuple[float, float]:
        """
        Calculates precision and recall.

        Parameters
        ----------
        pred_associations_batch
            list of predicted associations for each image pair
        gt_associations_batch
            list of ground truth associations for each image pair

        Returns
        -------
        values
            precision and recall
        """
        if not equally_sized(
            [
                pred_associations_batch,
                gt_associations_batch,
            ]
        ):
            raise ValueError(
                "GT and predicted associations batches must be the same size"
            )
        total_tp = 0
        total_fp = 0
        total_gt_size = 0

        for pred_associations, gt_associations in tzip(
            pred_associations_batch, gt_associations_batch
        ):
            if len(pred_associations) != 0:
                tp = self.indicator.indicate(pred_associations, gt_associations).sum()
                total_tp += tp
                total_fp += len(pred_associations) - tp
            gt = len(gt_associations)
            total_gt_size += gt

        recall = total_tp / total_gt_size if total_gt_size != 0 else 0
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp != 0 else 0

        return precision, recall


def precision_recall_fscore(
    pred_associations_batch: List[ArrayNx2[float]],
    gt_associations_batch: List[ArrayNx2[float]],
) -> Tuple[float, float, float]:
    """
    Calculates precision, recall, and F-score in a line association problem.

    Parameters
    ----------
    pred_associations_batch
        list of predicted associations for each image pair
    gt_associations_batch
        list of ground truth associations for each image pair

    Returns
    -------
    values
        precision, recall, and fscore

    Notes
    -----
    A pair of line indices represent each association,
    the first index corresponds to a line in the line array for the first image,
    and the second index corresponds to a line in the line array for the second image.

    Examples
    --------
    >>> import numpy as np
    >>> pred_associations_batch = [np.array([[0, 0], [1, 1], [2, 2]])]
    >>> gt_associations_batch = [np.array([[0, 0], [1, 1], [3, 3]])]
    >>> precision_, recall_, fscore_ = precision_recall_fscore(
    >>>     pred_associations_batch,
    >>>     gt_associations_batch
    >>> )
    """
    precision, recall = PrecisionRecall().calculate(
        pred_associations_batch, gt_associations_batch
    )
    fscore = (
        2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    )

    return precision, recall, fscore


def precision(
    pred_associations_batch: List[ArrayNx2[float]],
    gt_associations_batch: List[ArrayNx2[float]],
) -> float:
    """
    Calculates precision in a line association problem.

    Parameters
    ----------
    pred_associations_batch
        list of predicted associations for each image pair
    gt_associations_batch
        list of ground truth associations for each image pair

    Returns
    -------
    value
        precision

    Notes
    -----
    A pair of line indices represent each association,
    the first index corresponds to a line in the line array for the first image,
    and the second index corresponds to a line in the line array for the second image.

    Examples
    --------
    >>> import numpy as np
    >>> pred_associations_batch = [np.array([[0, 0], [1, 1], [2, 2]])]
    >>> gt_associations_batch = [np.array([[0, 0], [1, 1], [3, 3]])]
    >>> precision_ = precision(
    >>>     pred_associations_batch,
    >>>     gt_associations_batch
    >>> )
    """
    precision, _ = PrecisionRecall().calculate(
        pred_associations_batch, gt_associations_batch
    )

    return precision


def recall(
    pred_associations_batch: List[ArrayNx2[float]],
    gt_associations_batch: List[ArrayNx2[float]],
) -> float:
    """
    Calculates recall in a line association problem.

    Parameters
    ----------
    pred_associations_batch
        list of predicted associations for each image pair
    gt_associations_batch
        list of ground truth associations for each image pair

    Returns
    -------
    value
        recall

    Notes
    -----
    A pair of line indices represent each association,
    the first index corresponds to a line in the line array for the first image,
    and the second index corresponds to a line in the line array for the second image.

    Examples
    --------
    >>> import numpy as np
    >>> pred_associations_batch = [np.array([[0, 0], [1, 1], [2, 2]])]
    >>> gt_associations_batch = [np.array([[0, 0], [1, 1], [3, 3]])]
    >>> recall_ = recall(
    >>>     pred_associations_batch,
    >>>     gt_associations_batch
    >>> )
    """
    _, recall = PrecisionRecall().calculate(
        pred_associations_batch, gt_associations_batch
    )

    return recall


def fscore(
    pred_associations_batch: List[ArrayNx2[float]],
    gt_associations_batch: List[ArrayNx2[float]],
) -> float:
    """
    Calculates F-score in a line association problem.

    Parameters
    ----------
    pred_associations_batch
        list of predicted associations for each image pair
    gt_associations_batch
        list of ground truth associations for each image pair

    Returns
    -------
    value
        F-score

    Notes
    -----
    A pair of line indices represent each association,
    the first index corresponds to a line in the line array for the first image,
    and the second index corresponds to a line in the line array for the second image.

    Examples
    --------
    >>> import numpy as np
    >>> pred_associations_batch = [np.array([[0, 0], [1, 1], [2, 2]])]
    >>> gt_associations_batch = [np.array([[0, 0], [1, 1], [3, 3]])]
    >>> recall_ = fscore(
    >>>     pred_associations_batch,
    >>>     gt_associations_batch
    >>> )
    """
    _, _, fscore = precision_recall_fscore(
        pred_associations_batch, gt_associations_batch
    )

    return fscore
