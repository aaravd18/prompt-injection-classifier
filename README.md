
# Prompt Injection Classifier

A lightweight binary classifier that detects prompt injection attempts in chatbot applications using a fine-tuned DeBERTa transformer.

## Motivation

As someone deeply involved in building LLM-based chatbots, I’ve become increasingly interested in the security risks they face, especially with prompt injection attacks. With growing interest in cybersecurity, I wanted to build a classifier that can help preemptively detect and block such attacks before they ever reach an LLM call.

## Model & Architecture

This project fine-tunes a [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) transformer encoder on the [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) dataset.

- **Architecture**: Transformer encoder + MLP classification head
- **Objective**: Binary classification — Standard Prompt vs Injection
- **Tokenizer**: DeBERTa V3 tokenizer (slow mode fallback for compatibility)

## Results

- **Test Accuracy**: **91.6%**
- Model demonstrates strong performance in detecting diverse injection styles.

## 🛠️ Training Details

| Hyperparameter     | Value           |
|--------------------|-----------------|
| Learning Rate      | `2e-5`          |
| Train Batch Size   | `8`             |
| Eval Batch Size    | `8`             |
| Epochs             | `3`             |
| Optimizer          | `Adam` with betas=(0.9, 0.999), epsilon=1e-8 |
| Scheduler          | Linear          |

Training was performed using Hugging Face’s `Trainer` API and evaluated with the built-in `accuracy` metric.

## Main Implementation Highlights

- Preprocessed raw text using Hugging Face tokenizers with truncation and padding.
- Added an optional cell to freeze early transformer layers for faster training, commented out when training on Google Colab with GPU.

## References

- https://huggingface.co/deepset/deberta-v3-base-injection
- https://huggingface.co/datasets/deepset/prompt-injections
- https://medium.com/data-science/fine-tuning-bert-for-text-classification-a01f89b179fc
