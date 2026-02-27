"""Shared plotting helpers for consistent project-wide figure styling."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    "primary": "#1f5aa6",
    "primary_light": "#dbe8f6",
    "accent": "#d04a2b",
    "accent_light": "#f7ddd5",
    "secondary": "#2f7d4a",
    "secondary_light": "#ddefe4",
    "neutral": "#5f6b7a",
    "neutral_light": "#eef2f6",
    "highlight": "#8a63c7",
    "highlight_light": "#ebe2f8",
    "text": "#1f2933",
}

MODEL_NAME_MAP = {
    "XGBRegressor": "XGB",
    "GradientBoosting": "GBDT",
    "RandomForestRegressor": "RF",
    "RandomForest": "RF",
    "MLPRegressor": "MLP",
}


def apply_plot_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#d4dbe3",
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#ccd6e0",
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#d4dbe3",
        }
    )


def style_axis(ax, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)


def add_stat_box(
    ax,
    text: str,
    *,
    loc: str = "upper right",
    facecolor: str | None = None,
    fontsize: int = 10,
) -> None:
    positions = {
        "upper right": (0.98, 0.95, "right", "top"),
        "upper left": (0.02, 0.95, "left", "top"),
        "lower right": (0.98, 0.05, "right", "bottom"),
        "lower left": (0.02, 0.05, "left", "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=facecolor or "white",
            edgecolor="#d4dbe3",
            alpha=0.96,
        ),
    )


def format_model_name(name: str) -> str:
    return MODEL_NAME_MAP.get(name, name)


def plot_top_barh(
    ax,
    labels,
    values,
    *,
    title: str,
    xlabel: str,
    top_n: int | None = None,
    base_color: str | None = None,
    highlight_top: int = 3,
) -> None:
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    if top_n is not None:
        labels = labels[:top_n]
        values = values[:top_n]
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = values[order]

    colors = [base_color or COLORS["highlight"]] * len(values)
    for i in range(1, min(highlight_top, len(colors)) + 1):
        colors[-i] = COLORS["accent"] if i == 1 else COLORS["highlight"]

    bars = ax.barh(labels, values, color=colors, alpha=0.92)
    x_max = max(float(values.max()), 1e-12)
    ax.set_xlim(0, x_max * 1.18)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    style_axis(ax, grid_axis="x")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + x_max * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )


def plot_metric_hist(
    ax,
    values,
    *,
    title: str,
    xlabel: str,
    real_value: float | None = None,
    p_value: float | None = None,
    stats_text: str | None = None,
    color: str | None = None,
) -> None:
    values = np.asarray(values, dtype=float)
    ax.hist(values, bins=25, color=color or COLORS["secondary"], edgecolor="white", alpha=0.9)
    if real_value is not None:
        ax.axvline(
            real_value,
            color=COLORS["accent"],
            linewidth=2.2,
            linestyle="--",
            label=f"Real model ({real_value:.4f})",
        )
    ax.axvline(0, color=COLORS["neutral"], linewidth=1.2, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    suffix = f" (p={p_value:.4f})" if p_value is not None else ""
    ax.set_title(f"{title}{suffix}")
    style_axis(ax, grid_axis="y")
    if real_value is not None:
        ax.legend(loc="upper left", fontsize=9)
    if stats_text:
        add_stat_box(ax, stats_text, loc="upper right", facecolor=COLORS["neutral_light"], fontsize=9)


def plot_model_comparison(ax, names, cv_r2, test_r2) -> None:
    short_names = [format_model_name(name) for name in names]
    cv_r2 = np.asarray(cv_r2, dtype=float)
    test_r2 = np.asarray(test_r2, dtype=float)
    x_pos = np.arange(len(short_names))
    width = 0.34

    best_cv = int(np.argmax(cv_r2))
    best_test = int(np.argmax(test_r2))

    cv_colors = [COLORS["primary_light"]] * len(short_names)
    test_colors = [COLORS["accent_light"]] * len(short_names)
    cv_colors[best_cv] = COLORS["primary"]
    test_colors[best_test] = COLORS["accent"]

    bars1 = ax.bar(x_pos - width / 2, cv_r2, width, label="CV Val R2", color=cv_colors, edgecolor="none")
    bars2 = ax.bar(x_pos + width / 2, test_r2, width, label="Test R2", color=test_colors, edgecolor="none")

    min_score = min(min(cv_r2), min(test_r2), 0.0)
    max_score = max(max(cv_r2), max(test_r2), 0.0)
    pad = max((max_score - min_score) * 0.12, 0.05)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=0)
    ax.set_ylabel("R2")
    ax.set_title("Model Comparison")
    ax.set_ylim(min_score - pad, max_score + pad)
    ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=1)
    style_axis(ax, grid_axis="y")
    ax.legend(loc="upper right", fontsize=9)

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.008 if height >= 0 else -0.025),
            f"{height:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )

    add_stat_box(
        ax,
        f"Best CV: {short_names[best_cv]}\nBest Test: {short_names[best_test]}",
        loc="lower right",
        facecolor=COLORS["neutral_light"],
        fontsize=9,
    )
