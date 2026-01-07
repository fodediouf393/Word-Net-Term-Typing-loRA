from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path
import json

import pandas as pd
from datasets import Dataset

from term_typing.constants import LABEL2ID
from term_typing.features import FeatureBuilder


def load_table(path: str) -> pd.DataFrame:
    """Charge JSONL/JSON/CSV de façon robuste.
    - Vérifie l'existence du fichier
    - Ignore les lignes vides en JSONL
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"File not found: {p.resolve()}\n"
            f"Tip: run training from the repo root, or use absolute paths in configs/data.yaml."
        )

    suffix = p.suffix.lower()

    if suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {i} in {p}: {e}") from e
        if len(rows) == 0:
            raise ValueError(f"JSONL file is empty (or only blank lines): {p}")
        return pd.DataFrame(rows)

    if suffix == ".json":
        return pd.read_json(p)

    if suffix == ".csv":
        return pd.read_csv(p)

    raise ValueError(f"Unsupported file format: {p} (expected .jsonl, .json, .csv)")


def build_hf_dataset(
    df: pd.DataFrame,
    term_field: str,
    context_field: str,
    label_field: str,
    feature_builder: FeatureBuilder,
    require_labels: bool,
) -> Dataset:
    """Construit un HuggingFace Dataset.

    - require_labels=True : on exige la présence de label_field (train)
    - require_labels=False: label_field peut être absent (test), on met label=-1
    """
    if term_field not in df.columns:
        raise KeyError(f"Missing term_field '{term_field}' in columns: {list(df.columns)}")

    # context peut être absent => on ajoute une colonne vide
    if context_field not in df.columns:
        df[context_field] = ""

    # label peut être absent en test
    if require_labels and (label_field not in df.columns):
        raise KeyError(f"Missing label_field '{label_field}' in columns: {list(df.columns)}")

    def row_to_example(row) -> Dict:
        term = str(row[term_field]) if row[term_field] is not None else ""
        context = str(row[context_field]) if row[context_field] is not None else ""

        text = feature_builder.build_text(term=term, context=context)

        ex = {
            "text": text,
            "term": term,
            "context": context,
        }

        # label si disponible
        if label_field in df.columns:
            label = str(row[label_field]).strip().lower()
            if label not in LABEL2ID:
                raise ValueError(f"Unknown label '{label}'. Expected one of: {list(LABEL2ID.keys())}")
            ex["label"] = LABEL2ID[label]
            ex["label_str"] = label
        else:
            # test sans labels
            ex["label"] = -1
            ex["label_str"] = None

        # garder ID si présent
        if "ID" in df.columns:
            ex["ID"] = row["ID"]

        return ex

    examples = [row_to_example(row) for _, row in df.iterrows()]
    return Dataset.from_list(examples)


def load_datasets(
    train_path: str,
    test_path: str,
    term_field: str,
    context_field: str,
    label_field: str,
    feature_builder: FeatureBuilder,
) -> Tuple[Dataset, Dataset]:
    train_df = load_table(train_path)
    test_df = load_table(test_path)

    train_ds = build_hf_dataset(
        train_df, term_field, context_field, label_field, feature_builder, require_labels=True
    )
    test_ds = build_hf_dataset(
        test_df, term_field, context_field, label_field, feature_builder, require_labels=False
    )

    return train_ds, test_ds
