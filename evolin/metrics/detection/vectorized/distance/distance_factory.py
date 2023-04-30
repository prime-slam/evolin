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

from enum import Enum

from evolin.metrics.detection.vectorized.distance.distance import Distance
from evolin.metrics.detection.vectorized.distance.orthogonal import OrthogonalDistance
from evolin.metrics.detection.vectorized.distance.structural import StructuralDistance


class DistanceName(Enum):
    orthogonal = 0
    structural = 1

    @staticmethod
    def to_string(delimiter: str = ", "):
        return delimiter.join(dist.name for dist in DistanceName)


class DistanceFactory:
    @staticmethod
    def from_string(distance_name: str) -> Distance:
        distances = {
            DistanceName.orthogonal: OrthogonalDistance,
            DistanceName.structural: StructuralDistance,
        }

        try:
            return distances[DistanceName[distance_name]]()
        except KeyError:
            raise ValueError(
                f"Unsupported metric {distance_name}. "
                f"Expected: {DistanceName.to_string()}"
            )
