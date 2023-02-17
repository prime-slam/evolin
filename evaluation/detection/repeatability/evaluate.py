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

from evaluation.common.parser import create_base_parser
from evaluation.detection.repeatability.evaluator import (
    ScoredEvaluator,
    UnscoredEvaluator,
)
from evaluation.detection.repeatability.geometry.io import read_poses
from evaluation.detection.repeatability.geometry.transform import (
    make_homogeneous_matrix,
)
from evaluation.common.metric_information.metric_info import MetricInfo
from evaluation.common.utils import read_csv_batch, write_metrics


def calculate_repeatability_metrics(
    pred_lines_batch_path: Path,
    depth_maps_batch_path: Path,
    poses_path: Path,
    associations_path: Path,
    calibration_matrix_path: Path,
    distance_thresholds: List[float],
    frames_steps: List[int],
    scores_batch_path: Optional[Path] = None,
    score_thresholds_path: Optional[Path] = None,
) -> List[MetricInfo]:
    use_scores = scores_batch_path is not None

    associations = np.genfromtxt(associations_path, dtype=int)
    images_index = associations[..., 0]
    depth_maps_index = associations[..., 1]

    pred_lines_batch = read_csv_batch(pred_lines_batch_path, index=images_index)
    scores_batch = (
        read_csv_batch(scores_batch_path, index=images_index) if use_scores else None
    )
    score_thresholds = (
        np.genfromtxt(score_thresholds_path).tolist() if use_scores else None
    )
    depth_maps_paths = sorted(depth_maps_batch_path.iterdir())
    depth_maps_paths = [depth_maps_paths[i] for i in depth_maps_index]

    calibration_matrix = make_homogeneous_matrix(np.genfromtxt(calibration_matrix_path))
    euclidean_transforms = read_poses(poses_path)

    evaluator = (
        ScoredEvaluator(
            pred_lines_batch,
            depth_maps_paths,
            euclidean_transforms,
            calibration_matrix,
            frames_steps,
            scores_batch,
            score_thresholds,
            distance_thresholds,
        )
        if use_scores
        else UnscoredEvaluator(
            pred_lines_batch,
            depth_maps_paths,
            euclidean_transforms,
            calibration_matrix,
            frames_steps,
            distance_thresholds,
        )
    )

    return evaluator.evaluate()


if __name__ == "__main__":
    parser = create_base_parser(prog=f"python {Path(__file__).name}")

    parser.add_argument(
        "--depths",
        "-D",
        metavar="PATH",
        help="path to the folder with depth maps",
        required=True,
    )

    parser.add_argument(
        "--poses",
        "-P",
        metavar="PATH",
        help="path to the file with poses",
        required=True,
    )

    parser.add_argument(
        "--associations",
        "-a",
        metavar="PATH",
        help="path to the file with associations between images and depth maps",
        required=True,
    )

    parser.add_argument(
        "--calibration-matrix",
        "-c",
        metavar="PATH",
        help="path to the file with calibration matrix",
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
        "--frames-steps",
        "-f",
        metavar="SEQ",
        nargs="+",
        type=int,
        help="distance thresholds in pixels",
        default=[10],
    )

    parser.add_argument(
        "--output-file",
        "-O",
        metavar="STR",
        help="name of output file",
        default="repeatability_metrics.json",
    )

    args = parser.parse_args()

    results = calculate_repeatability_metrics(
        pred_lines_batch_path=Path(args.pred_lines),
        depth_maps_batch_path=Path(args.depths),
        poses_path=Path(args.poses),
        associations_path=Path(args.associations),
        calibration_matrix_path=Path(args.calibration_matrix),
        distance_thresholds=args.distance_thresholds,
        frames_steps=args.frames_steps,
        scores_batch_path=Path(args.scores) if args.scores else None,
        score_thresholds_path=Path(args.score_thresholds)
        if args.score_thresholds
        else None,
    )
    output_path = Path(args.output) / args.output_file
    write_metrics(output_path, results)
