# WordNet Term Typing (Encoder-only + LoRA)

Pipeline baseline + LoRA (PEFT) pour prédire le type lexical (noun/verb/adjective/adverb) sur WordNet (SubTask A.1).

- Training/Eval avec Hugging Face Trainer
- Logs TensorBoard
- Inference sur fichiers JSONL/CSV

## Quickstart
1) Installer deps: `pip install -r requirements.txt`
2) Mettre tes données dans `data/raw/train.jsonl` et `data/raw/test.jsonl`
3) Lancer:
   - Baseline: `python -m term_typing.train --config configs/baseline.yaml --data configs/data.yaml`
   - LoRA:     `python -m term_typing.train --config configs/lora.yaml --data configs/data.yaml`
4) TensorBoard: `tensorboard --logdir outputs/runs --port 6006`
