# Combined Dynamic Virtual Spatiotemporal Graph Mapping for Traffic Prediction

## Abstract
The continuous expansion of the urban construction scale has recently contributed to the demand for the dynamics of traffic intersections that are managed, making adaptive modellings become a hot topic. Existing deep learning methods are powerful to fit complex heterogeneous graphs. However, they still have drawbacks, which can be roughly classified into two categories, 1) spatiotemporal async-modelling approaches separately consider temporal and spatial dependencies, resulting in weak generalization and large instability while aggregating; 2) spatiotemporal sync-modelling is hard to capture long-term temporal dependencies because of the local receptive field. In order to overcome above challenges, a Combined Dynamic Virtual spatiotemporal Graph Mapping (CDVGM) is proposed in this work. The contributions are the following: 1) a dynamic virtual graph Laplacian (DV GL) is designed, which considers both the spatial signal passing and the temporal features simultaneously; 2) the Longterm Temporal Strengthen model (LT2S) for improving the stability of time series forecasting; Extensive experiments demonstrate that CDVGM has excellent performances of fast convergence speed and low resource consumption and achieves the current SOTA effect in terms of both accuracy and generalization.

## Install
* python >= 3.7.4
* pytorch >= 1.8.1
* numpy
* scipy
* sk-learn
* pandas
* tqdm

## Dataset
The dataset are all publicly available. If you have trouble to reformat the shape of the dataset, please send me email.

If you have any problem while installing, please don't hesitate to open an issue, or sand me email yingmingpu@gmail.com.

## Citation
```
@article{pu2022combined,
  title={Combined Dynamic Virtual Spatiotemporal Graph Mapping for Traffic Prediction},
  author={Pu, Yingming},
  journal={arXiv preprint arXiv:2210.00704},
  year={2022}
}
```
