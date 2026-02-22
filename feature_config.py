"""Unified feature configuration for training/validation scripts."""

from __future__ import annotations

from typing import Iterable

# Selected features (auto-updated by 遗传.py / 特征筛选.py)
SELECTED_FEATURE_COLS = [
    "qed_1",
    "BCUT2D_MWLOW_1",
    "BCUT2D_CHGLO_1",
    "PEOE_VSA8_1",
    "SMR_VSA6_1",
    "VSA_EState2_1",
    "NumRotatableBonds_1",
    "fr_ketone_1",
    "fr_ketone_Topliss_1",
    "MaxAbsEStateIndex_2",
    "MinAbsEStateIndex_2",
    "ExactMolWt_2",
    "MinAbsPartialCharge_2",
    "FpDensityMorgan1_2",
    "PEOE_VSA6_2",
    "PEOE_VSA7_2",
    "SlogP_VSA1_2",
    "VSA_EState5_2",
    "NumRotatableBonds_2",
    "fr_NH2_2",
    "Delta_HeavyAtomMolWt",
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
