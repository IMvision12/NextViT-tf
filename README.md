# NextViT-tf

This repository is about an implementation of the research paper "Next-ViT: Next Generation Vision Transformer for Efficient Deployment in
Realistic Industrial Scenarios" using `Tensorflow`.

In this paper authors propose a next generation vision Transformer for efficient deployment in realistic industrial scenarios, namely Next-ViT, which dominates both CNNs and ViTs from the perspective of latency/accuracy trade-off. In this work, the Next Convolution Block (NCB) and Next Transformer Block (NTB) are respectively developed to capture local and global information with deployment-friendly mechanisms. Then, Next Hybrid Strategy (NHS) is designed to stack NCB and NTB in an efficient hybrid paradigm, which boosts performance in various downstream tasks. Extensive experiments show that Next-ViT significantly outperforms existing CNNs, ViTs and CNN-Transformer hybrid architectures with respect to the latency/accuracy trade-off across various vision tasks.

<p align="center">
  <img src="https://github.com/IMvision12/NextViT-tf/blob/main/img/img1.png" title="graph">
</p>

# Model Architecture

<p align="center">
  <img src="https://github.com/IMvision12/NextViT-tf/blob/main/img/img2.png" title="arch">
</p>

# Usage:

Clone Github Repo: 

```bash
$ git clone https://github.com/IMvision12/NextViT-tf
$ cd nextvit_tf
```
Then import model

```py
import tensorflow as tf
from models.nextvit_tf import nextvit_base
model = nextvit_base(input_shape=(224, 224, 3), num_classes=100)
print(model.summary(expand_nested=True))
```

# References

[1] NextViT paper: https://arxiv.org/pdf/2207.05501v4

[2] Official NextViT Repo: https://github.com/bytedance/next-vit
