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
import numpy.typing as npt

from typing import Annotated, Literal, TypeVar

__all__ = ["ArrayNxM", "ArrayN", "ArrayNx2", "ArrayNx4", "ArrayNx2x2", "ArrayNxMx2"]

DType = TypeVar("DType", bound=np.generic)

ArrayNxM = Annotated[npt.NDArray[DType], Literal["N", "M"]]

ArrayNxMx2 = Annotated[npt.NDArray[DType], Literal["N", "M", 2]]

ArrayN = Annotated[npt.NDArray[DType], Literal["N"]]

ArrayNx2 = Annotated[npt.NDArray[DType], Literal["N", 2]]

ArrayNx4 = Annotated[npt.NDArray[DType], Literal["N", 4]]

ArrayNx2x2 = Annotated[npt.NDArray[DType], Literal["N", 2, 2]]
