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

from abc import abstractmethod, ABC
from dataclasses import dataclass


@dataclass
class AdditionalMetricInfo(ABC):
    def to_dict(self):
        return self.__dict__

    @abstractmethod
    def get_name(self):
        pass


@dataclass
class DistanceInfo(AdditionalMetricInfo):
    distance: str
    distance_threshold: float

    def get_name(self):
        return "distance_information"


@dataclass
class ScoreInfo(AdditionalMetricInfo):
    score_threshold: float

    def get_name(self):
        return "score_information"


@dataclass
class FramesStepInfo(AdditionalMetricInfo):
    step: float

    def get_name(self):
        return "frames_step_information"
