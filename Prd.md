Product Requirements Document: Split-Brain Memory Integration
1. Overview
This document outlines the requirements for integrating "Split-Brain" long-term memory capabilities into a Large Language Model (LLM) training repository. The goal is to separate the model's "memory" (storage of facts) from its "processing" (next-token prediction) by adding an auxiliary loss function on intermediate layers.

2. Core Objectives
Improve Long-Context Recall: Enhance the model's ability to retrieve information from long contexts without degrading short-term generation quality.
Layer-Specific Memorization: Target specific transformer layers (e.g., middle semantic layers) for memory storage.
Accurate Evaluation: Robustly measure recall capabilities using "needle-in-a-haystack" probes.
3. Feature Requirements
3.1 Training Pipeline
Goal: Fine-tune pre-trained models (e.g., Gemma 270M/1B) with an auxiliary memory loss.

Dataset Support:

Support for long-document datasets (PG19, WikiText-103).
Disk Caching: Implement pickle-based caching of tokenized datasets to prevent re-processing on every run.
Streaming/Chunking: Efficiently handle long sequences (up to 2048+ tokens) with sliding windows.
Loss Functions:

Primary Loss: Standard Next-Token Prediction (Cross-Entropy).
Auxiliary Memory Loss:
Simple L1: Reconstruction loss between current hidden state H_t and past hidden state H_{t-k} (with stop-gradient on past).
DINO / Self-Distillation: Cross-entropy between a "teacher" distribution (past) and "student" distribution (current) using softmax temperatures.
Configurable Weights: Ability to toggle memory loss (--no-memory-loss) and adjust its weight (--memory-weight).
Model Architecture Integration:

Hidden State Access: The model forward pass must support returning hidden states from efficient intermediate layers (not just the final layer).
Layer Selection: CLI argument (e.g., --memory-layer) to specify which layer index is used for the memory loss.
Infrastructure:

Hardware: Multi-GPU support (Pytorch DDP).
Logging: Integration with experiment trackers (WandB) for loss components (xent, mem_loss) and throughput.
Checkpointing: Rolling checkpoints with resume support.
3.2 Evaluation Suite
Goal: Quantify memory improvements versus baseline.

Needle-in-a-Haystack (NIHS) Probe:

Mechanism: Insert specific "needle" facts (random UUIDs/Cities) into long context "haystacks" (PG19 text).
Metric: Multiple-Choice Accuracy (4-way) rather than raw log-probability thresholds.
Depth Testing: Evaluate recall at various context depths (256, 512, 1024, 2048, 4096 tokens).
Architecture: A separate probe script (linear classifier or direct probability comparison) trained/evaluated on fixed model embeddings.
Standard Benchmarks:

Perplexity: Rolling perplexity evaluation on held-out sets (WikiText/PG19).
Downstream Tasks: Zero-shot evaluation on standard reasoning tasks (ARC-Challenge, HellaSwag) to ensure general capabilities aren't degraded.
4. Technical Specifications (Reference Implementation)
Component	Specification
Base Model	Gemma 3 (270M / 1B) PT
Framework	PyTorch
Context Length	1024 - 4096 tokens
Memory Window	~512 tokens (sliding)
Optimization	Adafactor / AdamW
5. Development Milestones
Baseline: Training script with standard causal LM loss and dataset loader.
Memory Loss: Implement auxiliary loss function and wire it to model outputs.
Probe: Build the NIHS evaluator to measure impact.
Optimization: Add dataset caching and multi-GPU support.
Refinement: Implement DINO loss and per-layer targeting.
