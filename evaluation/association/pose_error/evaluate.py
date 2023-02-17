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
import numpy as np

from pathlib import Path
from typing import List

from evaluation.association.pose_error.evaluator import PoseErrorEvaluator
from evaluation.common.geometry.io import read_poses
from evaluation.common.geometry.transform import (
    make_homogeneous_matrix,
)
from evaluation.common.metric_information.metric_info import MetricInfo
from evaluation.common.utils import read_csv_batch, write_metrics


def calculate_pose_error(
    lines_batch_path: Path,
    associations_batch_path: Path,
    depth_maps_batch_path: Path,
    poses_path: Path,
    depth_associations_path: Path,
    calibration_matrix_path: Path,
    pose_error_auc_thresholds: List[float],
) -> List[MetricInfo]:
    depth_associations = np.genfromtxt(depth_associations_path, dtype=int)
    images_index = depth_associations[..., 0]
    depth_maps_index = depth_associations[..., 1]

    lines_batch = read_csv_batch(lines_batch_path, index=images_index)
    associations_batch = read_csv_batch(associations_batch_path)
    associations_batch = list(map(lambda assoc: assoc.astype(int), associations_batch))
    depth_maps_paths = sorted(depth_maps_batch_path.iterdir())
    depth_maps_paths = [depth_maps_paths[i] for i in depth_maps_index]
    calibration_matrix = make_homogeneous_matrix(np.genfromtxt(calibration_matrix_path))
    gt_absolute_poses = read_poses(poses_path)

    evaluator = PoseErrorEvaluator(
        lines_batch,
        associations_batch,
        depth_maps_paths,
        gt_absolute_poses,
        calibration_matrix,
        pose_error_auc_thresholds,
    )

    return evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--lines",
        "-l",
        metavar="PATH",
        help="path to the folder with lines",
        required=True,
    )

    parser.add_argument(
        "--associations",
        "-a",
        metavar="PATH",
        help="path to the folder with line associations",
        required=True,
    )

    parser.add_argument(
        "--depths",
        "-d",
        metavar="PATH",
        help="path to the folder with depth maps",
        required=True,
    )

    parser.add_argument(
        "--poses",
        "-p",
        metavar="PATH",
        help="path to the file with poses",
        required=True,
    )

    parser.add_argument(
        "--depth-associations",
        "-A",
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
        "--pose-error-auc-thresholds",
        "-t",
        metavar="SEQ",
        nargs="+",
        type=float,
        help="thresholds in degrees for angular error auc calculation",
        default=[1.0, 3.0, 5.0, 10.0],
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
        default="pose_errors.json",
    )

    args = parser.parse_args()

    results = calculate_pose_error(
        lines_batch_path=Path(args.lines),
        associations_batch_path=Path(args.associations),
        depth_maps_batch_path=Path(args.depths),
        poses_path=Path(args.poses),
        depth_associations_path=Path(args.depth_associations),
        calibration_matrix_path=Path(args.calibration_matrix),
        pose_error_auc_thresholds=args.pose_error_auc_thresholds,
    )
    output_path = Path(args.output) / args.output_file
    write_metrics(output_path, results)
