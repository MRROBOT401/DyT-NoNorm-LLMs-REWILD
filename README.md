# Fine-Tuning LLMs Without Normalization Layers: A DyT-Based Approach Using RE-WILD

This repository contains the codebase, results, and plots for our final project in **ECE-GY 9143: High-Performance Machine Learning (HPML)** at NYU.

**Team:**

* Richard Zhong ([rhz2020@nyu.edu](mailto:rhz2020@nyu.edu))
* Gopala Krishna Abba ([ga2664@nyu.edu](mailto:ga2664@nyu.edu))

---

## Problem Overview

Post-training large LLMs is computationally expensive, and normalization layers like LayerNorm add complexity to training and inference. We investigate whether these layers can be replaced with a simpler alternative — **Dynamic Tanh (DyT)** — while maintaining performance.

---

## Key Contributions

* Replaced all `LayerNorm` layers in DistilGPT2 and Pythia with a learnable **Dynamic Tanh (DyT)** activation: `DyT(x) = tanh(\alpha x)`
* Integrated **LoRA (Low-Rank Adaptation)** via HuggingFace PEFT to enable parameter-efficient fine-tuning
* Explored:

  * Fully frozen DyT
  * **Selective unfreezing** of DyT layers
  * **Full supervised fine-tuning (SFT)**
* Fine-tuned and evaluated across **Alpaca**, **ShareGPT**, and **RE-WILD** datasets

---

## Experimental Setup

**Models:**

* DistilGPT2 (80M)
* Pythia 410M (limited due to memory)

**Frameworks:**

* HuggingFace Transformers
* PEFT (LoRA)
* Colab Pro and NYU HPC (A100)

**Datasets:**

* Alpaca: Small-scale instruction tuning (\~52k)
* ShareGPT: Medium-scale real dialogue (\~90k)
* RE-WILD: Open-ended QA (\~35k used due to constraints)

**Logged:**

* Training and validation loss per 500 steps
* Perplexity (RE-WILD only)
* Prompt response outputs
* Inference time (Vanilla vs DyT)

---

## Key Results

| Dataset  | DyT Val Loss | Vanilla Val Loss | Loss Gap |
| -------- | ------------ | ---------------- | -------- |
| Alpaca   | \~8.3        | \~1.5            | 🔺6.8    |
| ShareGPT | \~8.3        | \~2.3            | 🔺6.0    |
| RE-WILD  | \~8.3        | \~0.9            | 🔺7.4    |

* **Inference Time**: DyT = 77.05s, Vanilla = 77.46s → \~0.5% speedup
* **Prompt Quality**: DyT generates literal, unstructured completions; vanilla preserves instruction-following and formatting better

---

## Repository Structure

```
DyT-NoNorm-LLMs-REWILD/
├── notebooks/               # Jupyter notebooks for each experiment
├── scripts/                 # Training scripts (vanilla, DyT, selective unfreeze)
├── data_utils/              # Tokenizer, formatting, and dataset cleaning
├── results/                 # Raw loss logs and saved metrics
├── plots/                   # All graphs used in our report & slides
├── report/                  # Presentation slides (HPML_Presentation.pdf)
└── README.md
```

---

## How to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Train a DyT-modified DistilGPT2 model on RE-WILD:

```bash
python scripts/train_rewild_dyt_selective_unfreeze.py --epochs 3 --lr 2e-5 --batch_size 8
```

Run vanilla baseline on ShareGPT:

```bash
python scripts/train_sharegpt_vanilla.py --epochs 3
```

---

## Observations

* DyT struggles to generalize without normalization layers, especially on larger, diverse corpora like RE-WILD
* Selective unfreezing helps, but performance gap remains significant
* Vanilla DistilGPT2 shows clean convergence; DyT plateaus at high loss
* Full SFT improves DyT, but undermines PEFT advantages

---

## Slides & Report

* [📄 HPML Final Slides (PDF)](./report/Presentation.pdf)

---

## Future Work

* Try DyT with **LLaMA 3.2B** using larger batch sizes
* Evaluate DyT with alternative norm-replacement functions
* Integrate DyT into **quantized** or **sparsely activated** LLMs

---

## Contact

For questions or collaborations, reach out to:

* Richard Zhong: [rhz2020@nyu.edu](mailto:rhz2020@nyu.edu)
* Gopala Krishna Abba: [ga2664@nyu.edu](mailto:ga2664@nyu.edu)
