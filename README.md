# DyT-NoNorm-LLMs-REWILD

![Dynamic Tanh](https://img.shields.io/badge/Dynamic%20Tanh-DyT-blue.svg) ![DistilGPT2](https://img.shields.io/badge/DistilGPT2-Model-orange.svg) ![LoRA](https://img.shields.io/badge/LoRA-Optimization-green.svg)

Welcome to the **DyT-NoNorm-LLMs-REWILD** repository! This project focuses on enhancing the performance of DistilGPT2 by replacing LayerNorm with Dynamic Tanh (DyT). We evaluate our approach on various benchmarks, including RE-WILD, Alpaca, and ShareGPT.

For the latest releases, please visit our [Releases section](https://github.com/MRROBOT401/DyT-NoNorm-LLMs-REWILD/releases).

## Table of Contents

1. [Introduction](#introduction)
2. [Background](#background)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Evaluation](#evaluation)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)

## Introduction

In the world of deep learning, optimizing transformer models is crucial for improving performance and efficiency. This repository introduces a novel approach by integrating Dynamic Tanh into the DistilGPT2 architecture, replacing the conventional LayerNorm. Our modifications aim to enhance model adaptability and performance across various tasks.

## Background

### What is DistilGPT2?

DistilGPT2 is a smaller, faster, and lighter version of the original GPT-2 model. It retains most of the language understanding capabilities while being more efficient in terms of computation and memory usage. This makes it suitable for applications where resources are limited.

### Dynamic Tanh (DyT)

Dynamic Tanh is an activation function that adapts its behavior based on the input data. Unlike static activation functions, DyT can improve model learning by providing a more flexible response to varying input distributions. This adaptability can lead to better convergence and improved performance in various tasks.

### LoRA

Low-Rank Adaptation (LoRA) is a technique that allows for efficient fine-tuning of large language models. By introducing low-rank updates to the model weights, LoRA reduces the number of parameters that need to be trained, making it faster and more resource-efficient.

## Installation

To get started with DyT-NoNorm-LLMs-REWILD, follow these steps:

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.8 or higher
- Transformers library from Hugging Face
- Additional libraries: `numpy`, `scipy`, `matplotlib`

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/MRROBOT401/DyT-NoNorm-LLMs-REWILD.git
cd DyT-NoNorm-LLMs-REWILD
```

### Install Required Packages

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Download Pre-trained Models

To use the pre-trained DistilGPT2 model, you can download it directly from Hugging Face:

```bash
from transformers import DistilGPT2Tokenizer, DistilGPT2Model

tokenizer = DistilGPT2Tokenizer.from_pretrained('distilgpt2')
model = DistilGPT2Model.from_pretrained('distilgpt2')
```

### Run the Model

After setting up the environment, you can run the model using the following command:

```bash
python run_model.py
```

## Usage

### Training the Model

To train the model with Dynamic Tanh, use the following command:

```bash
python train.py --model_name distilgpt2 --activation_function dyT --epochs 10
```

You can adjust parameters like `--epochs` and `--batch_size` according to your needs.

### Evaluating the Model

After training, you can evaluate the model on various benchmarks:

```bash
python evaluate.py --model_name distilgpt2 --dataset rewild
```

This command will provide metrics on how well the model performs on the RE-WILD dataset.

## Evaluation

We evaluated our model on three primary benchmarks:

1. **RE-WILD**: This dataset tests the model's ability to understand and generate text based on real-world scenarios.
2. **Alpaca**: A dataset designed to challenge the model's reasoning capabilities.
3. **ShareGPT**: This dataset focuses on conversational AI, assessing how well the model can engage in dialogue.

### Results

Our experiments show that replacing LayerNorm with Dynamic Tanh leads to:

- Improved convergence rates during training.
- Better performance metrics across all evaluated datasets.
- Reduced resource consumption during fine-tuning.

## Contributing

We welcome contributions to enhance this project. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to reach out:

- Email: your_email@example.com
- GitHub: [MRROBOT401](https://github.com/MRROBOT401)

For the latest releases, please visit our [Releases section](https://github.com/MRROBOT401/DyT-NoNorm-LLMs-REWILD/releases).

Thank you for your interest in DyT-NoNorm-LLMs-REWILD!