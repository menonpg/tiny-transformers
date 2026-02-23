# Tiny Transformers Experiment

Exploring how small we can make transformers while still achieving useful accuracy on real tasks.

## Motivation

A recent paper showed that a **777-parameter transformer** can learn 10-digit addition with 99% accuracy through "grokking" — a sudden jump from memorization to generalization.

This proves transformers can learn **real algorithms**, not just memorize patterns. The model literally couldn't memorize (would need 10^17× more parameters).

## Experiments

### 1. Micro-ViT for MNIST (`notebooks/micro_vit_mnist.ipynb`)

**Goal:** Train the smallest Vision Transformer that achieves >95% accuracy on MNIST.

**Architecture:**
- ~4,000 parameters (vs 60K for LeNet-5)
- 7×7 patches (16 patches total)
- Embedding dim: 32
- Single attention layer, 2 heads
- No MLP block (saves params)

**Run in Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/menonpg/tiny-transformers/blob/main/notebooks/micro_vit_mnist.ipynb)

## Results

*Coming soon — will update after running experiments*

## References

- [Grokking Paper (Addition)](https://github.com/yhavinga/gpt-acc-jax)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Grokking: Generalization Beyond Overfitting](https://arxiv.org/abs/2201.02177)

## Blog Post

See: [The 1,000-Parameter Challenge: Minimal Transformers That Actually Work](https://blog.themenonlab.com/blog/tiny-transformers-challenge)
