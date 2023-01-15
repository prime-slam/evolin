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

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, List

from evaluation.common.metric_information.additional_info import (
    AdditionalMetricInfo,
)


@dataclass
class MetricInfo(ABC):
    name: str
    additional_information: Optional[List[AdditionalMetricInfo]]

    def to_dict(self):
        info = self.__dict__
        additional_info = info["additional_information"]
        if additional_info is not None:
            info["additional_information"] = {
                info.get_name(): info.to_dict() for info in additional_info
            }
        return info


@dataclass
class SimpleMetricInfo(MetricInfo):
    value: float


@dataclass
class CompositeMetricInfo(MetricInfo):
    values: Dict
