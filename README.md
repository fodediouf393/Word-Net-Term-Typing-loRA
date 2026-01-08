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
- a lexical term **L**
- an optional context sentence **S**

the goal is to predict the semantic type **T**.

<p align="center">
  <b>T ∈ { noun, verb, adjective, adverb }</b>
</p>

---

## Mathematical Formulation

We model Term Typing as a classification function:

<p align="center">
  <img alt="f_theta" src="https://latex.codecogs.com/svg.image?\dpi{140}\bf%20f_{\theta}:(S,L)\rightarrow%20T" />
</p>

where:
- **L** is a lexical term,
- **S** is an optional context sentence,
- **T** is the predicted semantic type,
- **fθ** denotes a parameterized model.

---

### Encoder-Based Models

For encoder-based approaches, the input pair (S, L) is converted to a textual sequence **x** and passed through an encoder:

<p align="center">
  <img alt="h=Encoder(x)" src="https://latex.codecogs.com/svg.image?\dpi{140}\bf%20h=\mathrm{Encoder}_{\theta}(x)" />
</p>

The encoder output is mapped to a distribution over types:

<p align="center">
  <img alt="p(T|S,L)=softmax(Wh+b)" src="https://latex.codecogs.com/svg.image?\dpi{140}\bf%20p(T\mid%20S,L)=\mathrm{softmax}(Wh&plus;b)" />
</p>

The predicted type is:

<p align="center">
  <img alt="T_hat=argmax p(T|S,L)" src="https://latex.codecogs.com/svg.image?\dpi{140}\bf%20\hat{T}=\arg\max_{T\in\mathcal{T}}%20p(T\mid%20S,L)" />
</p>

Training is performed by minimizing cross-entropy loss on labeled examples.
We compare:
- **full fine-tuning** (baseline)
- **LoRA** parameter-efficient fine-tuning

---

### Few-Shot LLM Formulation

In the few-shot setting, model parameters are not updated. A prompt **P** is built from a small labeled set:

<p align="center">
  <img alt="P={(Si,Li,Ti)}" src="https://latex.codecogs.com/svg.image?\dpi{140}\bf%20P=\{(S_i,L_i,T_i)\}_{i=1}^{k}" />
</p>

Given a new input (S, L), the LLM predicts:

<p align="center">
  <img alt="T_hat=LLM(P,S,L)" src="https://latex.codecogs.com/svg.image?\dpi{140}\bf%20\hat{T}=\mathrm{LLM}(P,S,L)" />
</p>

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

All figures are saved in `outputs/report_images/`.

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

