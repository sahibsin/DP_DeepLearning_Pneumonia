# Benchmarking Differentially Private Residual Networks for Medical Imagery (ICML'20 Workshop on Health Systems-HSYS)

![Made With python 3.8.2](https://img.shields.io/badge/Made%20with-Python%203.8.2-brightgreen)![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)![Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-green.svg)

### Abstract

In this paper we measure the effectiveness of e-Differential Privacy (DP) when applied to medical imaging. <br />
We compare two robust differential privacy mechanisms: Local-DP and DP-SGD and benchmark their performance when analyzing medical imagery records. <br />
We analyze the trade-off between the model's accuracy and the level of privacy it guarantees, and also take a closer look to evaluate how useful these theoretical privacy guarantees actually prove to be in the real world medical setting.

### Experimental Setup
The experiments discussed in this section used an 18-Layer Residual Network(ResNet) previously trained to achieve convergence on the ImageNet task. <br />
Input images passed to the deep neural network were scaled to 256 × 256 pixels, and normalized to 1. For performing the experiments we used Python 3.8.2 and PyTorch 1.4.0.

### Datasets and Architectures

| Dataset      |     ReseNet18      |
| ------------ | :----------------: |
| APTOS        | :heavy_check_mark: |
| Chest X-Rays | :heavy_check_mark: |

## 

### Paper & Poster
Paper: https://arxiv.org/pdf/2005.13099.pdf <br />
Poster: https://manifoldcomputing.com/hsys_poster/

### Citation
`@article{singh2020benchmarking, 
  title={Benchmarking Differentially Private Residual Networks for Medical Imagery},
  author={Sahib Singh and Harshvardhan Sikka and Sasikanth Kotti and Andrew Trask},
  journal={arXiv preprint arXiv:2005.13099},
  year={2020}
}`
