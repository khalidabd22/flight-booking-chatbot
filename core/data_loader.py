"""Load user data from uploaded files or pasted text.

Supported: .csv .tsv .txt .md .json .xlsx .xls .pdf  + pasted CSV / JSON / plain text.
Every loader returns a `Dataset` or raises DataLoadError with a friendly message.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.errors import DataLoadError

TABLE_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".json"}
TEXT_EXT = {".txt", ".md"}
PDF_EXT = {".pdf"}
SUPPORTED_EXT = TABLE_EXT | TEXT_EXT | PDF_EXT
ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


@dataclass
class Dataset:
    name: str
    df: pd.DataFrame | None = None
    text: str | None = None

    @property
    def kind(self) -> str:
        return "table" if self.df is not None else "text"

    def summary(self) -> str:
        if self.df is not None:
            return f"{self.name}: {len(self.df)} rows x {len(self.df.columns)} columns"
        return f"{self.name}: {len(self.text or '')} characters of text"


def _decode(raw: bytes) -> str:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataLoadError(f"'{name}' me koi data rows nahi mili.")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() or f"column_{i}" for i, c in enumerate(df.columns)]
    # make duplicate column names unique
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    df.columns = cols
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace({"nan": "", "None": ""})
    if df.empty:
        raise DataLoadError(f"'{name}' clean karne ke baad khali hai.")
    return df.reset_index(drop=True)


def _read_delimited(text: str, name: str, sep: str | None = None) -> pd.DataFrame:
    if sep is None:
        try:
            sep = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines="skip", engine="python")
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"'{name}' khali hai.") from exc
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"'{name}' ko CSV ki tarah padh nahi paye: {exc}") from exc
    return _clean_df(df, name)


def _read_json(text: str, name: str) -> Dataset:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"'{name}' valid JSON nahi hai (line {exc.lineno}).") from exc
    try:
        if isinstance(obj, list):
            return Dataset(name, df=_clean_df(pd.json_normalize(obj), name))
        if isinstance(obj, dict):
            for value in obj.values():  # {"records": [...]} style
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    return Dataset(name, df=_clean_df(pd.json_normalize(value), name))
            return Dataset(name, df=_clean_df(pd.json_normalize(obj), name))
    except DataLoadError:
        raise
    except Exception:  # noqa: BLE001 - fall back to plain text
        pass
    return Dataset(name, text=json.dumps(obj, indent=2, ensure_ascii=False))


def _read_pdf(raw: bytes, name: str) -> Dataset:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DataLoadError("PDF ke liye `pip install pypdf` chalayein.") from exc
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise DataLoadError(f"'{name}' password-protected PDF hai.")
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except DataLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"'{name}' PDF padh nahi paye: {exc}") from exc
    if not text:
        raise DataLoadError(f"'{name}' me text nahi mila (shayad scanned image PDF hai).")
    return Dataset(name, text=text)


def load_bytes(raw: bytes, filename: str, max_mb: int = 20) -> Dataset:
    name = Path(filename or "upload").name
    ext = Path(name).suffix.lower()
    if not raw:
        raise DataLoadError(f"'{name}' khali file hai.")
    if len(raw) > max_mb * 1024 * 1024:
        raise DataLoadError(f"'{name}' {max_mb} MB se badi hai.")
    if ext not in SUPPORTED_EXT:
        raise DataLoadError(f"'{ext or name}' format supported nahi. Use: {', '.join(sorted(SUPPORTED_EXT))}")

    if ext in (".xlsx", ".xls"):
        try:
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        except ImportError as exc:
            raise DataLoadError("Excel ke liye `pip install openpyxl` chalayein.") from exc
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(f"'{name}' Excel padh nahi paye: {exc}") from exc
        frames = [df.assign(_sheet=sheet) for sheet, df in sheets.items() if not df.empty]
        if not frames:
            raise DataLoadError(f"'{name}' ki sab sheets khali hain.")
        return Dataset(name, df=_clean_df(pd.concat(frames, ignore_index=True), name))
    if ext in PDF_EXT:
        return _read_pdf(raw, name)

    text = _decode(raw)
    if ext == ".csv":
        return Dataset(name, df=_read_delimited(text, name))
    if ext == ".tsv":
        return Dataset(name, df=_read_delimited(text, name, sep="\t"))
    if ext == ".json":
        return _read_json(text, name)
    if not text.strip():
        raise DataLoadError(f"'{name}' me koi text nahi hai.")
    return Dataset(name, text=text)


def load_path(path: str | Path, max_mb: int = 20) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"File nahi mili: {p.name}")
    return load_bytes(p.read_bytes(), p.name, max_mb=max_mb)


def _looks_like_table(text: str) -> bool:
    lines = [l for l in text.strip().splitlines() if l.strip()][:20]
    if len(lines) < 2:
        return False
    for sep in (",", "\t", ";", "|"):
        counts = [l.count(sep) for l in lines]
        if counts[0] >= 1 and all(c == counts[0] for c in counts):
            return True
    return False


def load_pasted(text: str, fmt: str = "auto", name: str = "pasted_data") -> Dataset:
    """fmt: auto | csv | json | text"""
    text = (text or "").strip()
    if not text:
        raise DataLoadError("Paste box khali hai.")
    fmt = (fmt or "auto").lower()
    if fmt == "json" or (fmt == "auto" and text[:1] in "[{"):
        try:
            return _read_json(text, name)
        except DataLoadError:
            if fmt == "json":
                raise
    if fmt == "csv" or (fmt == "auto" and _looks_like_table(text)):
        try:
            return Dataset(name, df=_read_delimited(text, name))
        except DataLoadError:
            if fmt == "csv":
                raise
    return Dataset(name, text=text)
