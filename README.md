# Transfer-Learning-Capabilities-of-Vision-Transformers-and-CNNs
Investigating Transfer Learning Capabilities of Vision Transformers and CNNs by Fine-Tuning a Single Trainable Block

![License](https://img.shields.io/github/license/mmahdibarghi/Transfer-Learning-Capabilities-of-Vision-Transformers-and-CNNs)

## Table of Contents

- [Introduction](#introduction)
- [Abstract](#abstract)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
  - [Dataset](#dataset)
  - [Models](#models)
  - [Fine-Tuning](#fine-tuning)
  - [Hyperparameters](#hyperparameters)
  - [Last Layers and Loss](#last-layers-and-loss)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Introduction

Recent advancements in the field of computer vision have seen the rise of transformer-based architectures, surpassing the state-of-the-art set by Convolutional Neural Networks (CNNs) in accuracy. However, these models are computationally expensive to train from scratch. This project investigates the transfer learning capabilities of vision transformers and CNNs by fine-tuning a single trainable block.

## Abstract

This project explores the transfer learning capabilities of vision transformers and CNNs by fine-tuning only the last trainable block of pre-trained models on CIFAR-10. We compare their performance in terms of accuracy and efficiency. Our results show that transformer-based architectures not only achieve higher accuracy but also do so with fewer parameters compared to CNNs.

## Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/mmahdibarghi/Transfer-Learning-Capabilities-of-Vision-Transformers-and-CNNs.git
cd investigating-transfer-learning
pip install -r requirements.txt
```

## Usage

To run the experiments, simply execute the Jupyter Notebook:

```bash
jupyter notebook Investigating_Transfer_Learning_Capabilities_of_Vision_Transformers_and_CNNs_by_Fine_Tuning_a_Single_Trainable_Block.ipynb
```

Ensure that the necessary datasets and pre-trained models are available in the specified directories.

## Experimental Setup

### Dataset

We use the CIFAR-10 dataset for fine-tuning the models pre-trained on ImageNet1K. CIFAR-10 contains 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images. The images are upscaled to 224x224 using bilinear interpolation for compatibility with pre-trained models.

### Models

We select a variety of CNN and transformer-based models, including:

- **CNNs**: DenseNet201
- **Transformers**: Vision Transformer  CaiT-S24, DeiTBaseDistilled

### Fine-Tuning

Fine-tuning is done by unfreezing only the last block of each model and training it on CIFAR-10. This approach helps evaluate the raw transfer learning capabilities of each architecture.

### Hyperparameters

- **Learning Rate**: 0.0001
- **Epochs**: 10
- **Batch Size**: 32 



## Results

The results of the experiments are summarized in the following table:

| Model             | Model type  | Trainable parameters | Validation accuracy (paper) | Validation accuracy (my results) | Train accuracy | Training time per epoch | Validation time per epoch |
|-------------------|-------------|----------------------|-----------------------------|---------------------------------|----------------|-------------------------|---------------------------|
| Densenet201       | CNN         | 235210               | 94.757                      | 91.44                          | 97.72          | 299.59s                 | 53.12s                    |
| DeiTBaseDistilled | transformer | 7103252              | 96.450                      | 96.44                          | 99.87          | 626.44s                 | 108.66s                   |
| CaiTS24           | transformer | 1775376              | 96.00                       | 96.78                          | 99.89          | 590.62s                 | 109.12s                   |


## Conclusion

The experiments conclude that transformer-based models generally outperform CNNs in terms of transfer learning efficiency and accuracy. Specifically, the DeiTBaseDistilled and CaiT-S24 models achieve the highest accuracy with fewer trainable parameters, highlighting the potential of transformer architectures in transfer learning tasks.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

1. Malpure, D., Litake, O. and Ingle, R., 2021. Investigating transfer learning capabilities of vision transformers and CNNs by fine-tuning a single trainable block. arXiv preprint arXiv:2110.05270.
2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
