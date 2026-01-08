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
- a **lexical term** \( L \),
- an optional **context sentence** \( S \),

the goal is to predict the **semantic type** \( T \) of the term.

In the WordNet setting, the type space is defined as:
\[
\mathcal{T} = \{\text{noun}, \text{verb}, \text{adjective}, \text{adverb}\}
\]

---

## Mathematical Formulation

Formally, the Term Typing task is modeled as a classification problem:

\[
f_\theta : (S, L) \;\longrightarrow\; T
\]

where:
- \( L \) is a lexical term,
- \( S \) is an optional context sentence,
- \( T \in \mathcal{T} \) is the predicted term type,
- \( f_\theta \) denotes a parameterized model.

---

### Encoder-Based Models

For encoder-based approaches, the input pair \((S, L)\) is transformed into a textual
representation \( x \), which is processed by a transformer encoder:

\[
h = \text{Encoder}_\theta(x)
\]

The encoder output is mapped to a probability distribution over types:
\[
p(T \mid S, L) = \text{softmax}(W h + b)
\]

The predicted type is:
\[
\hat{T} = \arg\max_{T \in \mathcal{T}} p(T \mid S, L)
\]

Training is performed by minimizing the cross-entropy loss on labeled examples.
Both **full fine-tuning** and **LoRA-based fine-tuning** are considered.

---

### Few-Shot LLM Formulation

In the few-shot setting, no model parameters are updated.
Instead, a prompt is constructed using a small set of labeled examples:

\[
P = \{(S_i, L_i, T_i)\}_{i=1}^{k}
\]

Given a new input \((S, L)\), the LLM predicts:
\[
\hat{T} = \text{LLM}(P, S, L)
\]

The output is constrained to belong to the predefined type set \(\mathcal{T}\).

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
fewshot_llms/
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


