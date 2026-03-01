"""Unified feature configuration for training/validation scripts."""

from __future__ import annotations

from typing import Iterable

# Selected features (auto-updated by 遗传.py / 特征筛选.py)
SELECTED_FEATURE_COLS = [
    "Delta_MaxAbsCharge",
    "VSA_EState8_2",
    "MinPartialCharge_1",
    "MaxAbsPartialCharge_1",
    "HallKierAlpha_2",
    "MolLogP_2",
    "Morgan_Similarity",
    "MaxAbsEStateIndex_1",
    "ExactMolWt_2",
    "InvT_x_DTPSA",
    "MaxAbsEStateIndex_2",
    "Delta_TPSA",
    "InvT_x_DMaxCharge",
    "Avalon_Similarity",
    "FpDensityMorgan1_2",
    "Inv_T",
    "Chi1v_1",
    "PEOE_VSA8_1",
    "Delta_LogP",
    "EState_VSA9_1",
    "PEOE_VSA6_1",
    "Kappa2_2",
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
