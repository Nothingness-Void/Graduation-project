"""Unified feature configuration for training/validation scripts."""

from __future__ import annotations

from typing import Iterable

# Selected features (auto-updated by 遗传.py / 特征筛选.py)
SELECTED_FEATURE_COLS = [
    "Delta_MaxAbsCharge",
    "MinPartialCharge_1",
    "MinPartialCharge_2",
    "MinEStateIndex_1",
    "MaxEStateIndex_1",
    "InvT_x_DTPSA",
    "HeavyAtomMolWt_1",
    "SMR_VSA1_1",
    "Delta_TPSA",
    "InvT_x_DMaxCharge",
    "FpDensityMorgan1_2",
    "PEOE_VSA8_2",
    "Inv_T",
    "Delta_LogP",
    "Chi2v_2",
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
