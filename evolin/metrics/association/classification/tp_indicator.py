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

import numpy as np

from evolin.typing import ArrayNx2, ArrayN


class TPIndicator:
    def indicate(
        self, pred_associations: ArrayNx2[int], gt_associations: ArrayNx2[int]
    ) -> ArrayN[bool]:
        if len(gt_associations.shape) == 1:
            gt_associations = gt_associations[None]
        if gt_associations.size == 0:
            return np.zeros(len(pred_associations), dtype=bool)
        m = gt_associations[:, np.newaxis] == pred_associations
        return m.all(axis=-1).any(axis=0)
