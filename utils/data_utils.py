# -*- coding: utf-8 -*-
"""
utils/data_utils.py - 共享数据工具函数

提供跨脚本复用的 train/test split 加载功能，
确保全链路（遗传 → 特征筛选 → Sklearn → DNN → Y-Randomization）使用一致的数据划分。
"""

import os
import numpy as np

# 默认的 split 索引文件路径
DEFAULT_SPLIT_INDEX_PATH = "results/train_test_split_indices.npz"


def load_saved_split_indices(
    n_samples: int,
    split_index_path: str = DEFAULT_SPLIT_INDEX_PATH,
):
    """
    Load train/test split indices from a .npz file if available and valid.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the current dataset. Used to verify
        consistency with the saved split.
    split_index_path : str
        Path to the .npz file containing split indices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        (train_idx, test_idx) if the file is valid, else None.
    """
    if not os.path.exists(split_index_path):
        return None
    try:
        with np.load(split_index_path, allow_pickle=False) as d:
            train_idx = d["train_idx"].astype(int)
            test_idx = d["test_idx"].astype(int)
            saved_n = int(d["n_samples"][0]) if "n_samples" in d else None
    except Exception:
        return None

    # Validate
    if saved_n is not None and saved_n != n_samples:
        return None
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None
    if np.any(train_idx < 0) or np.any(test_idx < 0):
        return None
    if np.any(train_idx >= n_samples) or np.any(test_idx >= n_samples):
        return None
    if len(np.unique(train_idx)) != len(train_idx):
        return None
    if len(np.unique(test_idx)) != len(test_idx):
        return None
    if np.intersect1d(train_idx, test_idx).size > 0:
        return None
    return train_idx, test_idx
