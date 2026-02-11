"""Unified feature configuration for training/validation scripts."""

from __future__ import annotations

from typing import Iterable

# Selected features from results/feature_ranking.txt (特征筛选结果)
# Selected features from Genetic Algorithm (R2 ~0.0268)
SELECTED_FEATURE_COLS = [
    'MolWt1', 
    'LabuteASA1', 
    'logP2', 
    'Delta_TPSA', 
    'HB_Match', 
    'Inv_T'
]

# Full feature set before feature selection
ALL_FEATURE_COLS = [
    "MolWt1",
    "logP1",
    "TPSA1",
    "MaxAbsPartialCharge1",
    "LabuteASA1",
    "MolWt2",
    "logP2",
    "TPSA2",
    "MaxAbsPartialCharge2",
    "LabuteASA2",
    "Avalon Similarity",
    "Morgan Similarity",
    "Topological Similarity",
    "Delta_LogP",
    "Delta_TPSA",
    "HB_Match",
    "Delta_MolMR",
    "CSP3_1",
    "CSP3_2",
    "Inv_T",
]


def resolve_target_col(columns: Iterable[str], preferred: str = "χ-result") -> str:
    """Resolve target column with fallback for encoding variations."""
    cols = list(columns)
    if preferred in cols:
        return preferred

    candidates = [c for c in cols if "result" in str(c).lower()]
    if candidates:
        return candidates[0]

    raise KeyError("未找到目标列：χ-result（或包含 result 的列名）")
