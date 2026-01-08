# WordNet Term Typing with Encoders, LoRA and Few-Shot LLMs

This project studies the **Term Typing task (Task A.1 – WordNet)** from the LLMs4OL framework.
The objective is to predict the **generalized lexical type** of a term using
encoder-based models and few-shot Large Language Models (LLMs).

We explore and compare two complementary paradigms:

1. **Encoder-based models**
   - BERT, RoBERTa, DistilBERT
   - Full fine-tuning (baseline)
   - Parameter-efficient fine-tuning with **LoRA**
2. **Few-shot prompting with LLMs**
   - Mistral
   - LLaMA (via Groq)
   - DeepSeek
   - Local small language models

The focus of this work is on **performance vs computational cost trade-offs**
and on clean, reproducible experimental comparisons.

---

## Task Definition

The task corresponds to **WordNet SubTask A.1 (Term Typing)**.

Given:
- a **lexical term** L,
- an optional **context sentence** S,

the goal is to predict the **semantic type** T of the term.

In the WordNet setting, the type space is defined as:

T ∈ {noun, verb, adjective, adverb}

---

## Mathematical Formulation

The Term Typing task is modeled as a classification function:

f(S, L) → T

where:
- L is a lexical term,
- S is an optional context sentence,
- T is the predicted semantic type.

---

### Encoder-Based Models

Given a pair (S, L), a textual input x is constructed and passed to a transformer encoder:

h = Encoder(x)

The encoder representation is mapped to a probability distribution over types:

p(T | S, L) = softmax(W · h + b)

The predicted type is obtained as:

T̂ = argmax p(T | S, L)

Training is performed using cross-entropy loss.
Both **full fine-tuning** and **parameter-efficient fine-tuning with LoRA** are considered.

---

### Few-Shot LLM Formulation

In the few-shot setting, model parameters remain frozen.
A prompt P is constructed from a small set of labeled examples:

P = {(S₁, L₁, T₁), …, (Sₖ, Lₖ, Tₖ)}

Given a new input (S, L), the LLM predicts:

T̂ = LLM(P, S, L)

The output is constrained to belong to the predefined type set.

---

## Project Structure

configs/ Training and LoRA configuration files
data/
├── raw/ WordNet datasets
└── processed/ Optional processed data
src/term_typing/
├── train.py Encoder training (baseline / LoRA)
├── predict.py Inference
├── eval_from_id.py Evaluation and confusion matrices
├── make_plots.py Comparative plots
└── run_sweep.py Full experimental sweep

├── few-shot_mistral.py
├── few-shot_llama.py
├── few-shot_deepseek.py
└── few-shot_sml.py
outputs/
├── runs/ Training logs and checkpoints
├── predictions/ Model predictions
├── metrics/ Evaluation metrics (JSON)
└── report_images/ Figures and confusion matrices


---

## Evaluation Metrics

- Accuracy
- Macro-F1 score
- Confusion matrices
- Training cost indicators (time, number of trainable parameters)

All figures are automatically saved in `outputs/report_images/`.

---

## Objectives

This project aims to answer the following questions:

- Can LoRA achieve performance comparable to full fine-tuning at a lower cost?
- How do encoder-based approaches compare to few-shot LLM prompting?
- What are the performance–efficiency trade-offs across model families?

---

## Reproducibility

- Fixed random seeds
- Explicit YAML configuration files
- TensorBoard logging for all encoder-based experiments
- Unified evaluation pipeline

---


