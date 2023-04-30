"""The line association problem can be thought of as a classification problem.
This submodule implements popular classification metrics in relation to the association problem."""

import evolin.metrics.association.classification.precision_recall_fscore_ as classification_module

from evolin.metrics.association.classification.precision_recall_fscore_ import *

__all__ = classification_module.__all__.copy()
