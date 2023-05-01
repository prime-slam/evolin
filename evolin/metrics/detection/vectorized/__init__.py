"""In the case of vector representation of lines,
we can classify individual lines and also calculate repeatability metrics.
This submodule implements the specified metrics."""

import evolin.metrics.detection.vectorized.average_precision as average_precision_module
import evolin.metrics.detection.vectorized.precision_recall_curve as precision_recall_curve_module
import evolin.metrics.detection.vectorized.precision_recall_fscore as precision_recall_fscore_module
import evolin.metrics.detection.vectorized.repeatability_localization_error as repeatability_localization_error_module

from evolin.metrics.detection.vectorized.distance.distance import Distance
from evolin.metrics.detection.vectorized.average_precision import *
from evolin.metrics.detection.vectorized.precision_recall_curve import *
from evolin.metrics.detection.vectorized.precision_recall_fscore import *
from evolin.metrics.detection.vectorized.repeatability_localization_error import *

__all__ = ["Distance"]
__all__ += average_precision_module.__all__
__all__ += precision_recall_curve_module.__all__
__all__ += precision_recall_fscore_module.__all__
__all__ += repeatability_localization_error_module.__all__
