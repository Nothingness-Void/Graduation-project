"""Unified feature configuration for training/validation scripts."""

from __future__ import annotations

from typing import Iterable

# Selected features (auto-updated by 遗传.py / 特征筛选.py)
SELECTED_FEATURE_COLS = [
    "HeavyAtomMolWt_1",
    "NumValenceElectrons_1",
    "MaxPartialCharge_1",
    "FpDensityMorgan3_1",
    "AvgIpc_1",
    "Chi1v_1",
    "Chi2n_1",
    "Chi2v_1",
    "Kappa1_1",
    "PEOE_VSA8_1",
    "SMR_VSA4_1",
    "SMR_VSA5_1",
    "SMR_VSA6_1",
    "EState_VSA9_1",
    "NumRotatableBonds_1",
    "fr_Ar_N_1",
    "fr_C_O_1",
    "fr_aldehyde_1",
    "MinAbsPartialCharge_2",
    "Chi3n_2",
    "SMR_VSA7_2",
    "SlogP_VSA6_2",
    "VSA_EState9_2",
    "NHOHCount_2",
    "Delta_LogP",
    "Delta_TPSA",
    "Delta_AromaticRings",
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

