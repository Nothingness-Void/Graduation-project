"""Unified feature configuration for training/validation scripts."""

from __future__ import annotations

from typing import Iterable

# Selected features (auto-updated by 遗传.py / 特征筛选.py)
SELECTED_FEATURE_COLS = [
    "BCUT2D_MWLOW_1",
    "PEOE_VSA4_1",
    "PEOE_VSA8_1",
    "SMR_VSA5_1",
    "SMR_VSA6_1",
    "SlogP_VSA3_1",
    "FractionCSP3_1",
    "fr_halogen_1",
    "fr_ketone_1",
    "fr_ketone_Topliss_1",
    "fr_methoxy_1",
    "MaxPartialCharge_2",
    "MinAbsPartialCharge_2",
    "FpDensityMorgan1_2",
    "Chi1_2",
    "Chi2n_2",
    "Chi3n_2",
    "SMR_VSA2_2",
    "SMR_VSA7_2",
    "SlogP_VSA6_2",
    "HeavyAtomCount_2",
    "NumAromaticCarbocycles_2",
    "NumRotatableBonds_2",
    "NumSaturatedCarbocycles_2",
    "MolLogP_2",
    "fr_Al_COO_2",
    "fr_NH2_2",
    "Delta_MolWt",
    "HB_Match",
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
