# WordNet Term Typing with Encoders, LoRA and Few-Shot LLMs

This project studies the **Term Typing task (Task A.1 – WordNet)** from the LLMs4OL framework.
The objective is to predict the **generalized lexical type** of a term using encoder-based models
and few-shot Large Language Models (LLMs).

We explore and compare two complementary paradigms:

1. **Encoder-based models**
   - BERT, RoBERTa, DistilBERT
   - Full fine-tuning (baseline)
   - Parameter-efficient fine-tuning with **LoRA**
2. **Few-shot prompting with LLMs**
   - Mistral
   - Qwen (local small language model)

The focus of this work is on **performance vs computational cost trade-offs**
and on clean, reproducible experimental comparisons.

---

## Task Definition

The task corresponds to **WordNet SubTask A.1 (Term Typing)**.

Given:
- a lexical term **L**
- an optional context sentence **S**

the goal is to predict the semantic type **T**, where:

$$
T \in \{\text{noun},\ \text{verb},\ \text{adjective},\ \text{adverb}\}
$$

---

## Mathematical Formulation

We model Term Typing as a classification function:

$$
f_{\theta}(S, L) \rightarrow T
$$

where:
- **L** is a lexical term  
- **S** is an optional context sentence  
- **T** is the predicted semantic type  
- **f\_{\theta}** denotes a parameterized model  

---

## Results

### Encoder-Based Models

#### Overall Performance and Cost

<p align="center">
  <img src="outputs_encoder/report_images/costs_table.png" width="900">
</p>

<p align="center">
  <em>Comparison of encoder-based models in baseline and LoRA settings, highlighting performance and computational cost.</em>
</p>

---

#### Macro-F1 by Backbone (Baseline vs LoRA)

<p align="center">
  <img src="outputs_encoder/report_images/perf_macro_f1_by_backbone.png" width="600">
</p>

<p align="center">
  <em>Macro-F1 comparison across backbones for full fine-tuning and LoRA.</em>
</p>

---

#### Confusion Matrices

<p align="center">
  <img src="outputs_encoder/report_images/confusion_bert-baseline-e5-20260107-215753.png" width="280">
  <img src="outputs_encoder/report_images/confusion_bert-lora-e5-20260107-215753.png" width="280">
</p>

<p align="center">
  <img src="outputs_encoder/report_images/confusion_distilbert-baseline-e5-20260107-215753.png" width="280">
  <img src="outputs_encoder/report_images/confusion_distilbert-lora-e5-20260107-215753.png" width="280">
</p>

<p align="center">
  <img src="outputs_encoder/report_images/confusion_roberta-baseline-e5-20260107-215753.png" width="280">
  <img src="outputs_encoder/report_images/confusion_roberta-lora-e5-20260107-215753.png" width="280">
</p>

<p align="center">
  <em>Confusion matrices for encoder-based models. Baseline models show strong diagonal dominance, while LoRA degrades class balance, especially for adjectives and adverbs.</em>
</p>

---

### Few-Shot LLMs

#### Performance Comparison (Mistral vs Qwen)

<p align="center">
  <img src="output_llms/report_images_llms/bar_mistral_vs_qwen_acc_and_macro_f1.png" width="550">
</p>

<p align="center">
  <em>Accuracy and Macro-F1 comparison between Mistral (24B) and Qwen2.5-0.5B in the few-shot setting.</em>
</p>

---

#### Confusion Matrices (LLMs)

<p align="center">
  <img src="output_llms/report_images_llms/confusion_mistral.png" width="320">
  <img src="output_llms/report_images_llms/confusion_qwen.png" width="320">
</p>

<p align="center">
  <em>Confusion matrices for few-shot LLMs. Larger models exhibit more stable predictions across lexical categories.</em>
</p>

---

## Project Structure

```text
configs/                  Training and LoRA configuration files
data/
  ├── raw/                WordNet datasets
  └── processed/          Optional processed data
src/term_typing/
  ├── train.py            Encoder training (baseline / LoRA)
  ├── predict.py          Inference
  ├── eval_from_id.py     Evaluation and confusion matrices
  ├── make_plots.py       Comparative plots
  └── run_sweep.py        Full experimental sweep
fewshot_llms/
  ├── few-shot_mistral.py
  ├── few-shot_llama.py
  ├── few-shot_deepseek.py
  └── few-shot_sml.py
outputs_encoder/
  └── report_images/      Encoder-based figures and confusion matrices
output_llms/
  └── report_images_llms/ LLM comparison figures
