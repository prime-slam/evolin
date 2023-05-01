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

import json
import numpy as np

from pathlib import Path
from typing import List, Iterable, Optional
from PIL import Image

from evaluation.common.metric_information.metric_info import MetricInfo
from evolin.typing import ArrayNx4, ArrayNx2, ArrayN


def read_csv_batch(
    batch_path: Path, delimiter=",", index: Optional[Iterable[int]] = None
) -> List[np.ndarray]:
    paths_batch = sorted(batch_path.iterdir())
    index = range(len(paths_batch)) if index is None else index
    batch = [np.genfromtxt(paths_batch[i], delimiter=delimiter) for i in index]
    return batch


def clip_lines(lines: ArrayNx4[float], height: float, width: float) -> ArrayNx4[float]:
    x_index = [0, 2]
    lines[..., x_index] = np.clip(lines[..., x_index], 0, width - 1)
    y_index = [1, 3]
    lines[..., y_index] = np.clip(lines[..., y_index], 0, height - 1)
    return lines


def scale_lines(
    lines: ArrayNx4[float], x_scaler: float, y_scaler: float
) -> ArrayNx4[float]:
    x_index = [0, 2]
    lines[..., x_index] *= x_scaler
    y_index = [1, 3]
    lines[..., y_index] *= y_scaler
    return lines


def is_nonzero_length(lines: ArrayNx4[float]) -> ArrayN[bool]:
    return np.logical_and.reduce(lines[..., [0, 1]] != lines[..., [2, 3]], axis=-1)


def filter_lines_by_score(
    lines_batch: List[ArrayNx4[float]],
    scores_batch: List[ArrayN[float]],
    threshold: float,
) -> List[ArrayNx4[float]]:
    return [
        lines[scores > threshold] for lines, scores in zip(lines_batch, scores_batch)
    ]


def create_image_sizes_batch(images_path: Path) -> ArrayNx2[float]:
    # PIL allows read the size without allocating memory for the image
    images = [Image.open(path) for path in sorted(images_path.iterdir())]
    sizes = np.array([[image.height, image.width] for image in images])
    for image in images:
        image.close()
    return sizes


def write_metrics(output: Path, metrics: List[MetricInfo]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    dicts = [metric_info.to_dict() for metric_info in metrics]
    with open(output, "w") as out:
        json.dump(dicts, out, ensure_ascii=False)
