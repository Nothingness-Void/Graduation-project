"""Unified feature configuration for training/validation scripts."""

from __future__ import annotations

from typing import Iterable

# Selected features (auto-updated by 遗传.py / 特征筛选.py)
SELECTED_FEATURE_COLS = [
    "MinAbsPartialCharge_1",
    "BCUT2D_MWLOW_1",
    "VSA_EState1_1",
    "HeavyAtomMolWt_2",
    "SMR_VSA5_2",
    "VSA_EState3_2",
    "Delta_MolMR",
    "Delta_MaxAbsCharge",
    "Inv_T",
]


def resolve_target_col(columns: Iterable[str], preferred: str = "chi_result") -> str:
    """Resolve target column with fallback for encoding variations."""
    cols = list(columns)
    if preferred in cols:
        return preferred

    candidates = [c for c in cols if "result" in str(c).lower()]
    if candidates:
        return candidates[0]

    raise KeyError("未找到目标列：chi_result（或包含 result 的列名）")
