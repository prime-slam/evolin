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
from scipy.spatial.transform import Rotation as R

from evaluation.detection.repeatability.geometry.transform import (
    make_euclidean_transform,
)


def read_poses(poses_path: Path, comment_symbol: str = "#"):
    euclidean_transforms = []
    for line in poses_path.read_text().splitlines():
        line = line.strip()
        if line.startswith(comment_symbol):
            continue
        _, tx, ty, tz, qx, qy, qz, qw = line.split(" ")
        translation = np.array([tx, ty, tz], dtype=float)
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        euclidean_transforms.append(
            np.linalg.inv(make_euclidean_transform(rotation, translation))
        )
    return euclidean_transforms
