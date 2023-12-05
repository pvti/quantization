# Quantization
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
![GitHub last commit](https://img.shields.io/github/last-commit/pvtien96/quantization)
![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2Fpvtien96%2Fquantization&label=visitors&countColor=%23263759&style=flat)

# Research Papers on Quantization for Network Compression

<details>
<summary><a href="https://arxiv.org/abs/2311.12023" target="_blank"><strong>1. LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning</strong></a></summary>

- **TLDR:** Decompose a matrix into low-rank component and a memory-efficient quantized component.
- **Abstract:** We propose a simple approach for memory-efficient adaptation of pretrained language models. Our approach uses an iterative algorithm to decompose each pretrained matrix into a high-precision low-rank component and a memory-efficient quantized component. During finetuning, the quantized component remains fixed and only the low-rank component is updated. We present an integer linear programming formulation of the quantization component which enables dynamic configuration of quantization parameters (e.g., bit-width, block size) for each matrix given an overall target memory budget. We further explore a data-aware version of the algorithm which uses an approximation of the Fisher information matrix to weight the reconstruction objective during matrix decomposition. Experiments on adapting RoBERTa and LLaMA-2 (7B and 70B) demonstrate that our low-rank plus quantized matrix decomposition approach (LQ-LoRA) outperforms strong QLoRA and GPTQ-LoRA baselines and moreover enables more aggressive quantization. For example, on the OpenAssistant benchmark LQ-LoRA is able to learn a 2.5-bit LLaMA-2 model that is competitive with a model finetuned with 4-bit QLoRA. When finetuned on a language modeling calibration dataset, LQ-LoRA can also be used for model compression; in this setting our 2.75-bit LLaMA-2-70B model (which has 2.85 bits on average when including the low-rank components and requires 27GB of GPU memory) is competitive with the original model in full precision.
- **What:** 
- **Methodology:**
- **Conclusions:**
- **Limitations:**
- **Comments:** First paper.

</details>
