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

from pathlib import Path
from typing import List, Optional

from evaluation.detection.classification.heatmap.evaluator import (
    ScoredEvaluator,
    UnscoredEvaluator,
)
from evaluation.common.metric_information.metric_info import (
    MetricInfo,
)
from evaluation.common.parser import create_base_parser
from evaluation.common.utils import (
    read_csv_batch,
    create_image_sizes_batch,
    write_metrics,
)


def calculate_heatmap_metrics(
    images_path: Path,
    pred_lines_batch_path: Path,
    gt_lines_batch_path: Path,
    scores_batch_path: Optional[Path] = None,
    score_thresholds_path: Optional[Path] = None,
) -> List[MetricInfo]:
    use_scores = scores_batch_path is not None

    gt_lines_batch = read_csv_batch(gt_lines_batch_path)
    pred_lines_batch = read_csv_batch(pred_lines_batch_path)
    scores_batch = read_csv_batch(scores_batch_path) if use_scores else None
    score_thresholds = np.genfromtxt(score_thresholds_path) if use_scores else None
    image_sizes_batch = create_image_sizes_batch(images_path)
    heights_batch = image_sizes_batch[..., 0]
    widths_batch = image_sizes_batch[..., 1]

    evaluator = (
        ScoredEvaluator(
            pred_lines_batch,
            gt_lines_batch,
            scores_batch,
            heights_batch,
            widths_batch,
            score_thresholds,
        )
        if use_scores
        else UnscoredEvaluator(
            pred_lines_batch, gt_lines_batch, heights_batch, widths_batch
        )
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
        "--output-file",
        "-O",
        metavar="STR",
        help="name of output file",
        default="heatmap_metrics.json",
    )

    args = parser.parse_args()
    results = calculate_heatmap_metrics(
        images_path=Path(args.imgs),
        pred_lines_batch_path=Path(args.pred_lines),
        gt_lines_batch_path=Path(args.gt_lines),
        scores_batch_path=Path(args.scores) if args.scores else None,
        score_thresholds_path=Path(args.score_thresholds)
        if args.score_thresholds
        else None,
    )

    output_path = Path(args.output) / args.output_file
    write_metrics(output_path, results)
