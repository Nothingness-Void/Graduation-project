"""Unified feature configuration for training/validation scripts."""

from __future__ import annotations

from typing import Iterable

# Selected features (auto-updated by 遗传.py / 特征筛选.py)
SELECTED_FEATURE_COLS = [
    "qed_1",
    "MaxPartialCharge_1",
    "MinPartialCharge_1",
    "Chi0n_1",
    "SMR_VSA5_1",
    "FractionCSP3_1",
    "fr_C_O_1",
    "fr_ketone_1",
    "fr_ketone_Topliss_1",
    "MaxPartialCharge_2",
    "Chi1n_2",
    "Chi2n_2",
    "PEOE_VSA9_2",
    "SMR_VSA10_2",
    "SlogP_VSA5_2",
    "TPSA_2",
    "EState_VSA2_2",
    "MolLogP_2",
    "Delta_LabuteASA",
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
