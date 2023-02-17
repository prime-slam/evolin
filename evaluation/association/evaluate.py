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

import argparse

from pathlib import Path
from typing import List

from evaluation.association.evaluator import ClassificationEvaluator
from evaluation.common.metric_information.metric_info import (
    MetricInfo,
)
from evaluation.common.utils import (
    read_csv_batch,
    write_metrics,
)


def calculate_classification_metrics(
    pred_associations_batch_path: Path,
    gt_associations_batch_path: Path,
) -> List[MetricInfo]:
    gt_associations_batch = read_csv_batch(gt_associations_batch_path)
    pred_associations_batch = read_csv_batch(pred_associations_batch_path)

    evaluator = ClassificationEvaluator(
        gt_associations_batch,
        pred_associations_batch,
    )

    return evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pred-associations",
        "-p",
        metavar="PATH",
        help="path to the folder with predicted associations",
        required=True,
    )

    parser.add_argument(
        "--gt-associations",
        "-g",
        metavar="PATH",
        help="path to the folder with ground truth associations",
        required=True,
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="output path",
        required=True,
    )

    parser.add_argument(
        "--output-file",
        "-O",
        metavar="STR",
        help="name of output file",
        default="classification_metrics.json",
    )

    args = parser.parse_args()
    results = calculate_classification_metrics(
        pred_associations_batch_path=Path(args.pred_associations),
        gt_associations_batch_path=Path(args.gt_associations),
    )

    output_path = Path(args.output) / args.output_file
    write_metrics(output_path, results)
