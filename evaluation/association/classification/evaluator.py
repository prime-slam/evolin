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

from typing import List

from evaluation.common.evaluator_base import Evaluator
from evaluation.common.metric_information.metric_info import (
    MetricInfo,
    SimpleMetricInfo,
)
from src.metrics.association.classification import precision_recall_fscore
from src.typing import ArrayNx4


class ClassificationEvaluator(Evaluator):
    def __init__(
        self,
        pred_associations_batch: List[ArrayNx4[float]],
        gt_associations_batch: List[ArrayNx4[float]],
    ):
        self.pred_associations_batch = pred_associations_batch
        self.gt_associations_batch = gt_associations_batch

    def evaluate(self) -> List[MetricInfo]:
        precision, recall, fscore = precision_recall_fscore(
            self.pred_associations_batch,
            self.gt_associations_batch,
        )
        return [
            SimpleMetricInfo(
                name="precision", value=precision, additional_information=None
            ),
            SimpleMetricInfo(name="recall", value=recall, additional_information=None),
            SimpleMetricInfo(name="fscore", value=fscore, additional_information=None),
        ]
