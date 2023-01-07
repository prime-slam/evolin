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

import pytest

from src.metrics.detection.vectorized.distance.distance_factory import DistanceFactory
from src.metrics.detection.vectorized.distance.orthogonal import OrthogonalDistance
from src.metrics.detection.vectorized.distance.structural import StructuralDistance


@pytest.mark.parametrize(
    "distance_name, expected_class",
    [
        ("orthogonal", OrthogonalDistance),
        ("structural", StructuralDistance),
    ],
)
def test_distance_from_string(distance_name, expected_class):
    distance = DistanceFactory.from_string(distance_name)
    assert isinstance(distance, expected_class)


def test_unknown_distance():
    with pytest.raises(ValueError):
        DistanceFactory.from_string("unknown")
