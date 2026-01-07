from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from datasets import Dataset

from term_typing.constants import LABEL2ID
from term_typing.features import FeatureBuilder


def load_table(path: str) -> pd.DataFrame:
    """Charge JSONL/CSV automatiquement."""
    if path.lower().endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.lower().endswith(".json"):
        return pd.read_json(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def build_hf_dataset(
    df: pd.DataFrame,
    term_field: str,
    context_field: str,
    label_field: str,
    feature_builder: FeatureBuilder,
) -> Dataset:
    # Normaliser colonnes
    if term_field not in df.columns:
        raise KeyError(f"Missing term_field '{term_field}' in columns: {list(df.columns)}")
    if label_field not in df.columns:
        raise KeyError(f"Missing label_field '{label_field}' in columns: {list(df.columns)}")

    # context peut ne pas exister
    if context_field not in df.columns:
        df[context_field] = ""

    def row_to_example(row) -> Dict:
        term = str(row[term_field]) if row[term_field] is not None else ""
        context = str(row[context_field]) if row[context_field] is not None else ""
        label = str(row[label_field]).strip().lower()
        if label not in LABEL2ID:
            raise ValueError(f"Unknown label '{label}'. Expected one of: {list(LABEL2ID.keys())}")

        text = feature_builder.build_text(term=term, context=context)
        return {
            "text": text,
            "label": LABEL2ID[label],
            "term": term,
            "context": context,
            "label_str": label,
        }

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

    train_ds = build_hf_dataset(train_df, term_field, context_field, label_field, feature_builder)
    test_ds = build_hf_dataset(test_df, term_field, context_field, label_field, feature_builder)

    return train_ds, test_ds
