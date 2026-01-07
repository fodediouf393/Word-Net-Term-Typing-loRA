from __future__ import annotations

LABELS = ["noun", "verb", "adjective", "adverb"]

LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}
