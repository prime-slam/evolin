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

from typing import Optional, List
from pathlib import Path

from evaluation.common.metric_information.metric_info import MetricInfo
from evaluation.common.parser import create_base_parser
from evaluation.common.utils import (
    read_csv_batch,
    clip_lines,
    is_nonzero_length,
    create_image_sizes_batch,
    scale_lines,
    write_metrics,
)
from evaluation.classification.vectorized.evaluator import (
    ScoredEvaluator,
    UnscoredEvaluator,
)
from src.metrics.detection.vectorized import (
    EVALUATION_RESOLUTION,
)
from src.typing import ArrayN


def calculate_vectorized_metrics(
    images_path: Path,
    pred_lines_batch_path: Path,
    gt_lines_batch_path: Path,
    distance_thresholds: ArrayN[float],
    scores_batch_path: Optional[Path] = None,
    score_thresholds_path: Optional[Path] = None,
) -> List[MetricInfo]:
    use_scores = scores_batch_path is not None

    gt_lines_batch = read_csv_batch(gt_lines_batch_path)
    pred_lines_batch = read_csv_batch(pred_lines_batch_path)
    scores_batch = read_csv_batch(scores_batch_path) if use_scores else None
    score_thresholds = np.genfromtxt(score_thresholds_path) if use_scores else None
    image_sizes_batch = create_image_sizes_batch(images_path)

    for i in range(len(gt_lines_batch)):
        height, width = image_sizes_batch[i]
        pred_lines = pred_lines_batch[i]
        gt_lines = gt_lines_batch[i]

        x_scaler = EVALUATION_RESOLUTION / width
        y_scaler = EVALUATION_RESOLUTION / height
        scale_lines(pred_lines, x_scaler, y_scaler)
        pred_lines = clip_lines(
            pred_lines, EVALUATION_RESOLUTION, EVALUATION_RESOLUTION
        )
        scale_lines(gt_lines, x_scaler, y_scaler)
        nonzero_length = is_nonzero_length(pred_lines)
        pred_lines_batch[i] = pred_lines[nonzero_length]
        if use_scores:
            scores_batch[i] = scores_batch[i][nonzero_length]

    evaluator = (
        ScoredEvaluator(
            pred_lines_batch,
            gt_lines_batch,
            scores_batch,
            score_thresholds,
            distance_thresholds,
        )
        if use_scores
        else UnscoredEvaluator(pred_lines_batch, gt_lines_batch, distance_thresholds)
    )

    return evaluator.evaluate()


if __name__ == "__main__":
    parser = create_base_parser(prog=f"python {Path(__file__).name}")

    parser.add_argument(
        "--imgs", "-i", metavar="PATH", help="path to images", required=True
    )

    parser.add_argument(
        "--gt-lines",
        "-g",
        metavar="PATH",
        help="path to the folder with ground truth lines",
        required=True,
    )

    parser.add_argument(
        "--distance-thresholds",
        "-d",
        metavar="SEQ",
        nargs="+",
        type=float,
        help="distance thresholds in pixels",
        default=[5.0, 10.0, 15.0],
    )

    parser.add_argument(
        "--output-file",
        "-O",
        metavar="STR",
        help="name of output file",
        default="vectorized_classification_metrics.json",
    )

    args = parser.parse_args()

    results = calculate_vectorized_metrics(
        images_path=Path(args.imgs),
        pred_lines_batch_path=Path(args.pred_lines),
        gt_lines_batch_path=Path(args.gt_lines),
        distance_thresholds=np.array(args.distance_thresholds),
        scores_batch_path=Path(args.scores) if args.scores else None,
        score_thresholds_path=Path(args.score_thresholds)
        if args.score_thresholds
        else None,
    )
    output_path = Path(args.output) / args.output_file
    write_metrics(output_path, results)
