"""Line segments can be represented in vector form, i.e., endpoints, and raster form,
in the form of a set of points, making it possible to evaluate two types of
detection metrics: heatmap and vectorized."""
from evolin.metrics.detection import vectorized
from evolin.metrics.detection import heatmap
