"""Unified feature configuration for training/validation scripts."""

from __future__ import annotations

from typing import Iterable

# Selected features (auto-updated by 遗传.py / 特征筛选.py)
SELECTED_FEATURE_COLS = [
    "Delta_MaxAbsCharge",
    "MinPartialCharge_1",
    "SPS_2",
    "Chi1n_1",
    "EState_VSA1_2",
    "MolWt_2",
    "InvT_x_DTPSA",
    "Chi2v_1",
    "Kappa1_2",
    "MaxAbsEStateIndex_2",
    "Inv_T",
    "Delta_LogP",
    "Delta_TPSA",
    "InvT_x_DMaxCharge",
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
