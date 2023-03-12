<div align="center">
  <img src="https://raw.githubusercontent.com/prime-slam/evolin/main/assets/logo.png">
</div>

---
[![tests](https://github.com/prime-slam/evolin/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/prime-slam/evolin/actions/workflows/ci.yml)
[![License: Apache License 2.0](https://img.shields.io/github/license/saltstack/salt)](https://opensource.org/license/apache-2-0/)

EVOLIN is a benchmark for evaluation of line detection and association results. We provide a set of docker-packed line detection and association algorithms, metrics to evaluate them, and line-annotated data.
Additional information can be found on our [web page](https://prime-slam.github.io/evolin/) and in the [article](https://arxiv.org/abs/2303.05162).

## Installation

### Dependencies

1. Install dependencies
```bash
sudo apt update \
&& sudo apt upgrade \
&& sudo apt install --no-install-recommends -y libeigen3-dev cmake
```
2. Install our custom `g2opy`:
```bash
git clone https://github.com/anastasiia-kornilova/g2opy
cd g2opy
git checkout lines_opt
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install
```
3. Clone this repository
```bash
git clone https://github.com/prime-slam/evolin
```

## Annotated data
To evaluate line detectors and associators,
we annotated `lr kt2` and `of kt2` trajectories from [ICL NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html),
as well as `fr3/cabinet` and `fr1/desk` trajectories from [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).
Only breaking segments have been annotated,
such as ceilings, floors, walls, doors, and furniture linear elements.
The datasets can be downloaded [here](https://drive.google.com/drive/folders/1pEqOfkScEPq9GgrvNFVzpEkI0PYKVxfm?usp=sharing).

## Metrics

The following detection metrics are implemented:
* Heatmap-based and vectorized classification
  * precision
  * recall
  * F-score
  * average precision
* Repeatability
  * repeatability score
  * localization error

The following association metrics are implemented:
* Matching classification
  * precision
  * recall
  * F-score
* Pose error
  * angular translation error
  * absolute translation error
  * angular rotation error
  * pose error AUC

## Get detection and association results
A list of algorithms and instructions for running them can be found in our [repository](https://github.com/prime-slam/line-detection-association-dockers).

## Evaluation
The scripts required for evaluation and examples, as well as the documentation are located in `evaluation` folder.
The results of the evaluation of adapted detection and association algorithms can be found in our [article](https://arxiv.org/abs/2303.05162).
## Cite us
If you find this work useful in your research, please consider citing:
```bibtex
@article{evolin2023,
title={EVOLIN Benchmark: Evaluation of Line Detection and Association},
author={Kirill Ivanov, Gonzalo Ferrer, and Anastasiia Kornilova},
journal={arXiv preprint arXiv:2303.05162},
year={2023}}
```
