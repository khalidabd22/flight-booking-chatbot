"""Turn loaded datasets into a compact text context for the LLM.

No vector DB needed: simple keyword scoring picks the most relevant rows / text chunks,
plus a schema + numeric summary so the model understands the whole table.
"""
from __future__ import annotations

import re

import pandas as pd

from core.data_loader import Dataset

_WORD = re.compile(r"[a-zA-Z0-9ऀ-ॿ]{2,}")
STOPWORDS = {
    "the", "is", "are", "a", "an", "of", "to", "in", "for", "on", "and", "or", "me", "my", "i",
    "what", "which", "how", "show", "tell", "about", "with", "kya", "hai", "ka", "ki", "ke", "ko",
    "se", "mein", "me", "batao", "please", "kaun", "kitna", "kitne", "this", "that", "do", "does",
}


def tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "") if w.lower() not in STOPWORDS}


def _table_context(ds: Dataset, query: str, max_rows: int, budget: int) -> str:
    df = ds.df
    parts = [f"### Table: {ds.name} ({len(df)} rows)", "Columns: " + ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)]

    numeric = df.select_dtypes("number")
    if not numeric.empty:
        stats = numeric.agg(["count", "sum", "mean", "min", "max"]).round(2)
        parts.append("Numeric summary:\n" + stats.to_csv())

    if len(df) <= max_rows:
        selected = df
        parts.append("All rows:")
    else:
        q = tokens(query)
        if q:
            row_text = df.astype(str).agg(" ".join, axis=1).str.lower()
            scores = row_text.apply(lambda t: sum(1 for w in q if w in t))
            top = scores[scores > 0].sort_values(ascending=False).head(max_rows).index
            selected = df.loc[top] if len(top) else df.head(max_rows)
        else:
            selected = df.head(max_rows)
        parts.append(f"Most relevant {len(selected)} of {len(df)} rows:")
    parts.append(selected.to_csv(index=False))
    return "\n".join(parts)[:budget]


def _text_context(ds: Dataset, query: str, budget: int, chunk_size: int = 800) -> str:
    text = ds.text or ""
    header = f"### Document: {ds.name}\n"
    if len(text) <= budget - len(header):
        return header + text
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    q = tokens(query)
    ranked = sorted(range(len(chunks)), key=lambda i: -sum(1 for w in q if w in chunks[i].lower()))
    picked, used = [], len(header)
    for i in ranked:
        if used + len(chunks[i]) > budget:
            break
        picked.append(i)
        used += len(chunks[i])
    return header + "\n...\n".join(chunks[i] for i in sorted(picked))


def build_context(datasets: list[Dataset], query: str, max_chars: int = 12000, max_rows: int = 40) -> str:
    if not datasets:
        return ""
    per = max(1500, max_chars // len(datasets))
    blocks = []
    for ds in datasets:
        try:
            if ds.kind == "table":
                blocks.append(_table_context(ds, query, max_rows, per))
            else:
                blocks.append(_text_context(ds, query, per))
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not break the chat
            blocks.append(f"### {ds.name}: (context banate waqt error: {exc})")
    return "\n\n".join(blocks)[:max_chars]


def find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Case-insensitive fuzzy column lookup, e.g. find_column(df, 'amount', 'value')."""
    lower = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for cand in candidates:
        for key, original in lower.items():
            if cand in key:
                return original
    return None
