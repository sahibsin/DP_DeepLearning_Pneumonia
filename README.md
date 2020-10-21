# Benchmarking Differentially Private Residual Networks for Medical Imagery

### Abstract
In this paper we measure the effectiveness of e-Differential Privacy (DP) when applied to medical imaging. 
We compare two robust differential privacy mechanisms: Local-DP and DP-SGD and benchmark their performance when analyzing medical imagery records. 
We analyze the trade-off between the model's accuracy and the level of privacy it guarantees, and also take a closer look to evaluate how useful these theoretical privacy guarantees actually prove to be in the real world medical setting.

### Experimental Setup
The experiments discussed in this section used an 18-Layer Residual Network(ResNet) previously trained to achieve convergence on the ImageNet task. Input images passed to the deep neural network were scaled to 256 × 256 pixels, and normalized to 1. For performing the experiments we used Python 3.8.2 and PyTorch 1.4.0.
