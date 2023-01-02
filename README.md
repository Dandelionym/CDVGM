<h2 align="center">Combined Dynamic Virtual Spatiotemporal Graph Mapping</h2>
<br>
<h5 align="center">This <a href="https://www.researchgate.net/publication/364126322_Combined_Dynamic_Virtual_Spatiotemporal_Graph_Mapping_for_Traffic_Prediction?_sg%5B0%5D=mdpk3xtXzGI84gwfVM7_NgGYjYzdiZh6SPgxlZsxKMX-KhDKPR-EZ6VlHKb-qYn0UihjRDf9p1msAmKwWTnE5pdILyUktKQekBGfjc5G.EZF8WgrjPkDgQWwhDKYI3Gtk09nfRaoCr7sozP2RF-99sn1Y1N_8_cBTcVlVX0BNnyr4u7SItxl4oDCeX8eUnw">paper</a> is under reviewing, if you download it, give it a star please</h5>

<br>
<p align="center">
<img align="center" src="https://img.shields.io/badge/Fast-80%25-blue" />
<img align="center" src="https://img.shields.io/badge/Stable-90%25-green" />
<img align="center" src="https://img.shields.io/badge/Topology_free-100%25-red" />
</p>
<br>

![images](https://github.com/Dandelionym/CDVGM/blob/main/imgs/framework.png)

## Abstract
The continuous expansion of the urban construction scale has recently contributed to the demand for the dynamics of traffic intersections that are managed, making adaptive modellings become a hot topic. Existing deep learning methods are powerful to fit complex heterogeneous graphs. However, they still have drawbacks, which can be roughly classified into two categories, 1) spatiotemporal async-modelling approaches separately consider temporal and spatial dependencies, resulting in weak generalization and large instability while aggregating; 2) spatiotemporal sync-modelling is hard to capture long-term temporal dependencies because of the local receptive field. In order to overcome above challenges, a Combined Dynamic Virtual spatiotemporal Graph Mapping (CDVGM) is proposed in this work. The contributions are the following: 1) a dynamic virtual graph Laplacian (DV GL) is designed, which considers both the spatial signal passing and the temporal features simultaneously; 2) the Longterm Temporal Strengthen model (LT2S) for improving the stability of time series forecasting; Extensive experiments demonstrate that CDVGM has excellent performances of fast convergence speed and low resource consumption and achieves the current SOTA effect in terms of both accuracy and generalization.

## Prepare the environment easily
* python >= 3.7.4
* pytorch >= 1.8.1 (CUDA)
* numpy, scipy, sk-learn, pandas, tqdm
* other packages that related above.
Here is the steps:
```
conda create -n cdvgm python=3.7.4
conda activate cdvgm
conda install numpy scipy sk-learn pandas tqdm
```
Then, install pytroch by following [this](https://codeantenna.com/a/wGlXIGEs77) guidance step by step. 

Note that, our code is tested successfully on `Linux` and `Windows10` with any GPU with at least 6GB v-memory.


## About Dataset
> Make sure you have corrected the file name.<br/>
> First you should try `PEMSD8` & `PEMSD4`, then `PEMSD3` and `PEMSD7`.

## About Running
> Check the run.py with your custom configs (Don't care about all others). <br/>
> Then just ruuuuuuuuuuuun it!


## About training log
> The log will be saved at ./root/ dir.<br/>
> Besides, it supports `pandas` and `matplotlib` to analyze & visualization.

## Acknowledgment
* This repo is based on [DGCN](https://github.com/guokan987/DGCN) (Guo et al.), which is excellent, useful, and helpful, here faithfully thanks a lot.

## License
All rights reserved. Licensed under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International). The code is released for academic research use only.
