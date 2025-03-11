# FuzHDA
The implementation of "Fuzzy Domain Adaptation from Heterogeneous Source Teacher Models" in Python. 

Code for the TFS publication. The full paper can be found [here](https://doi.org/10.1109/TFUZZ.2025.3541001). 

The code is developed based on [LCFD](https://github.com/tntek/source-free-domain-adaptation).

## Contribution

- A heterogeneous multi-source-free domain adaptation method that integrates fuzzy rules and deep neural networks.
- Transfer the latent feature extraction into a semantic space via a logit learning system provided by fuzzy outputs
- Extract various types of data knowledge by leveraging prompt learning and uncovers rich external causal factors of self-supervision from pseudo-labels..

## Overview
Source training:
![Framework](https://github.com/el3518/DCA/blob/main/image/fuz-ht-Page-s.pdf)

Target training:
![Framework](https://github.com/el3518/DCA/blob/main/image/fuz-ht-Page-t.pdf)

## Setup
Ensure that you have PyTorch 1.12.1+cu113

## Dataset
You can find the datasets [here](https://github.com/jindongwang/transferlearning/tree/master/data).

## Usage
For source training:
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs_fuz.py --cfg "cfgs/office/source_fuz.yaml" SETTING.S 0

For black-box target training:
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs_fuz.py --cfg "cfgs/office/bb_target.yaml" SETTING.T 0

For target training:
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs_fuz.py --cfg "cfgs/office/ht_lcfd_fuz.yaml" SETTING.T 0

## Results

| Task  | R2Dl | R2We  | R2Am | Avg  | 
| ---- | ---- | ---- | ---- | ---- |
| FuzHDA  | 99.2  | 97.0  | 83.6 | 93.2 |


Please consider citing if you find this helpful or use this code for your research.

Citation
```
@article{li2025fuzzy,
  title={Fuzzy Domain Adaptation From Heterogeneous Source Teacher Models},
  author={Li, Keqiuyin and Lu, Jie and Zuo, Hua and Zhang, Guangquan},
  journal={IEEE Transactions on Fuzzy Systems},
  year={2025},
  publisher={IEEE}
}
