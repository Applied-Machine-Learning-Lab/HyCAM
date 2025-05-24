# HyCAM: Hybrid Contextual Attention Modulation

This repository implements the HyCAM framework, which has been accepted to CIKM 2025 Full Research Track with the title "Contextual Attention Modulation: Towards Efficient Multi-Task Adaptation in Large Language Models".

The HyCAM framework is designed for the efficient and effective multi-task adaptation of Large Language Models (LLMs). It includes the following key components:

### Key Components

1.  **Contextual Attention Modulation (CAM)**: A core mechanism to dynamically modulate self-attention representations.
2.  **Hybrid Architecture (HyCAM)**: Combines a shared, full-parameter CAM with multiple specialized CAMs.
3.  **Dynamic Routing with Load Balancing**: A soft-routing mechanism with Gumbel-Softmax and an auxiliary load-balancing loss to manage specialized CAM contributions.

> **Note**: HyCAM requires **custom modifications to Hugging Face’s `transformers` library** to enable global load-balancing loss computation.


### Core Modifications (HuggingFace Transformers)

To support load balancing across specialized CAMs, the `forward` method of LLM models (e.g., `LlamaModel`, `Qwen2Model`, `MistralModel`) must be edited with stacked **routing logits and probabilities** in `self_utils.py`.

The global load balancing loss is as below:

```python
def calculate_load_balancing_loss(all_router_logits, all_gumbel_probs):
    num_specialized_cams = all_router_logits.size(-1)
    all_router_logits_flat = all_router_logits.reshape(-1, num_specialized_cams)
    all_gumbel_probs_flat = all_gumbel_probs.reshape(-1, num_specialized_cams)

    # 1. Calculate softmax of router_logits for each token
    softmax_logits_b_k = F.softmax(all_router_logits_flat, dim=-1) 

    # 2. Calculate the first term: (1/B * sum_b p_b,k) for each expert k
    mean_gumbel_probs_per_expert_k = torch.mean(all_gumbel_probs_flat, dim=0) # Shape: (N_s)

    # 3. Calculate the second term: (1/B * sum_b softmax(l_b)_k) for each expert k
    mean_softmax_logits_per_expert_k = torch.mean(softmax_logits_b_k, dim=0) # Shape: (N_s)

    # 4. Multiply these two terms element-wise for each expert
    product_of_means_k = mean_gumbel_probs_per_expert_k * mean_softmax_logits_per_expert_k # Shape: (N_s)

    # 5. Sum these products over all specialized CAM modules (experts k)
    load_balance_loss = torch.sum(product_of_means_k) # Scalar

    return load_balance_loss

```

---

###  How to Use

#### 1. **Prepare The Dataset**

Follow the instructions in `main.py` to format the multi-task dataset.

#### 2. **Run Training**

Use the provided `run_model_base.sh` script. Ensure the custom arguments are set in your script.

### Reproducibility Details

We provide detailed descriptions of the foundation models, datasets, and hyperparameters used in our experiments within the paper.

#### Key Python Packages

Below are the key Python package versions required for our implementation:


* `Python == 3.9`
* `PyTorch == 2.1.0`
* `Transformers == 4.45.0`
* `DeepSpeed == 0.13.2`
* `numpy == 1.26.4`
* `PEFT == 0.14.0`
* `CUDA == 11.8` & `CUDNN == 8`  or `CANN 8.0.rc2` (For Ascend)

#### Evaluation Setup

For evaluation, we use the `evaluate` library along with its built-in BLEU and ROUGE implementations. Additionally, our approach relies on `nltk` and `rouge_score` for text-based metric calculations. The corresponding package versions are:

* `evaluate == 0.4.3`
* `nltk == 3.9.1`
* `rouge_score == 0.1.2`

#### Hyperparameter Settings

The main hyperparameter settings used in our experiments are:

* **Batch Size**: 2
* **Gradient Accumulation Steps**: 2
* **Learning Rate**: 2e-5
* **Optimizer**: AdamW
* **Weight Decay**: 0.05
* **Scheduler**: CosineAnnealingLR


#### Random Seed Control

To eliminate randomness in training, we use different random seeds across experiments. Our core experiments are conducted using seed *2333*, while in loss curve visualizations, we additionally use seeds *100*, *200*, *300*, *400*, and *500* to compute the mean and variance trend lines for better statistical reliability.



### Feedback and Collaboration

We welcome feedback and collaboration on the HyCAM framework and future works. Please feel free to submit issues or pull requests if you have suggestions or would like to contribute.