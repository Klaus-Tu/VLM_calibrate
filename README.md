# Calibrate VLMs

This codebase provides an official implementation for the paper: [An Empirical Study Into What Matters for Calibrating Vision-Language Models
]([https://arxiv.org/abs/2303.13251](https://arxiv.org/pdf/2402.07417)) at ICML 2024.

### Abstract
Vision–Language Models (VLMs) have emerged as the dominant approach for zero-shot recognition, adept at handling diverse scenarios and significant distribution changes. However, their deployment in risk-sensitive areas requires a deep understanding of their uncertainty estimation capabilities, a relatively uncharted area. In this study, we explore the calibration properties of VLMs across different architectures, datasets, and training strategies. In particular, we analyze the uncertainty estimation performance of VLMs when calibrated in one domain, label set or hierarchy level, and tested in a different one. Our findings reveal that while VLMs are not inherently calibrated for uncertainty, temperature scaling significantly and consistently improves calibration, even across shifts in distribution and changes in label set. Moreover, VLMs can be calibrated with a very small set of examples. Through detailed experimentation, we highlight the potential applications and importance of our insights, aiming for more reliable and effective use of VLMs in critical, real-world scenarios.

<figure class="image">
  <p align="center">
    <img src="comparison.PNG" width="60%" height="60%" />
  </p>
</figure>

## PyTorch Implementation

This repository contains:

- the Python implementation of temperature scaling and ECE error.
- the pipeline of calibrating VLMs.

Please follow the instruction below to install it and run the experiment demo.

### Prerequisites
* Please install [OpenCLIP](https://github.com/mlfoundations/open_clip) and [LAVIS](https://github.com/salesforce/LAVIS) to access different VLMs. 
* Please download datasets used in experiments (details are included in supplementary of the paper).

## Example of calibrating ViT-B/16 by OpenAI
1. Load extracted image features and compute logits
```python
   import torch
   import open_clip

   # Load image features of a calibration dataset
   # It should be of size (n, d), n is the number of images, and d is the feature dimension
   cal_feature = torch.load('/path/to/features/cifar10_val.pt')

   # Load pre-computed zero-shot class weights 
   cal_weights = torch.load('/path/to/features/cifar10_weights.pt')

   # Derive the logit scale from the pre-trained model (each model has its own logit scale)
   model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', device=device, pretrained='openai')
   scale = model.logit_scale.exp().detach().cpu()

   # Compute logits for the calibration set
   cal_logits = scale * cal_feature @ cal_weights
```

1. Calibration using temperature scaling
```python
   from calibration_methods.temperature_scaling import *
   # Load the labels of calibration dataset
   cal_labels = torch.load('/path/to/features/cifar10_labels.pt')

   # Calibration using temperature scaling
   temp = tune_temp(cal_logits, cal_labels, lower=0.01, upper=10)
```

2. Compute ECE before and after calibration
```python
   # Load image features and class weights for test datasets and compute the logits
   test_logits = scale * test_feature @ test_weights
   labels = torch.load('/path/to/features/imagenet_labels.pt')

   # Compute ECE before calibration
   acc_before, ece_before = ECE(test_logits, labels)
   
   # Compute ECE after calibration
   acc_after, ece_after = ECE(test_logits/temp, labels)

   # Temperature scaling does not change model performance
   print(acc_before==acc_after)
```

## Citation
 ```bibtex
@inproceedings{tu2024cali_vlm,
  title={An Empirical Study Into What Matters for Calibrating Vision–Language Models},
  author={Tu, Weijie and Deng, Weijian and Campbell, Dylan and Gould, Stephen and Gedeon, Tom},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```


## License
MIT
