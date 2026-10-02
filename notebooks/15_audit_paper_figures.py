#!/usr/bin/env python3
"""Generate the main validation figure from frozen analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"

MODEL_LABELS = {
    "lfm2.5:8b-a1b-q4_K_M": "LFM2.5",
    "granite4:7b-a1b-h": "Granite 4",
    "MichelRosselli/apertus:8b-instruct-2509-q4_k_m": "Apertus",
    "glm-4.7-flash:q4_K_M": "GLM-4.7",
    "gpt-oss:20b": "GPT-OSS",
    "olmo-3:7b-instruct-q4_K_M": "OLMo 3",
    "nemotron-3-nano:4b": "Nemotron 3",
    "gpt-5.6-sol": "Codex Sol",
    "gpt-5.6-terra": "Codex Terra",
    "gpt-5.6-luna": "Codex Luna",
}

TASK_MARKERS = {
    "work_policy": "o",
    "office_investment": "s",
    "hiring_geography": "^",
    "performance_policy": "D",
    "exception_policy": "P",
    "team_design": "X",
}

HOLDOUT_COLORS = {
    "lfm2.5:8b-a1b-q4_K_M": "#0072B2",
    "granite4:7b-a1b-h": "#D55E00",
    "MichelRosselli/apertus:8b-instruct-2509-q4_k_m": "#009E73",
}

CROSS_LABEL_OFFSETS = {
    "lfm2.5:8b-a1b-q4_K_M": (4, 3),
    "granite4:7b-a1b-h": (4, 9),
    "MichelRosselli/apertus:8b-instruct-2509-q4_k_m": (4, 4),
    "glm-4.7-flash:q4_K_M": (4, 4),
    "gpt-oss:20b": (4, 5),
    "olmo-3:7b-instruct-q4_K_M": (4, 5),
    "nemotron-3-nano:4b": (4, -10),
}


def load_json(name: str) -> dict:
    with (RESULTS / name).open() as handle:
        return json.load(handle)


def identity_axis(ax: plt.Axes, low: float, high: float) -> None:
    ax.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1)
    ax.axhline(0, color="#BBBBBB", linewidth=0.7)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.grid(color="#E6E6E6", linewidth=0.6)


def main() -> None:
    audit = load_json("susceptibility_audit_structured_analysis.json")
    cross = load_json("cross_interface_audit_analysis.json")
    frontier = load_json("codex_frontier_audit_analysis.json")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 180,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25), constrained_layout=True)

    ax = axes[0]
    for cell in audit["cells"]:
        if cell["role"] != "holdout":
            continue
        model = cell["model"]
        ax.scatter(
            cell["audit_effect"],
            cell["full_effect"],
            s=44,
            marker=TASK_MARKERS[cell["task"]],
            color=HOLDOUT_COLORS[model],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            label=MODEL_LABELS[model],
        )
    identity_axis(ax, -1.1, 0.5)
    ax.set_xlabel("Five-document audit effect")
    ax.set_ylabel("Disjoint 45-document feed effect")
    ax.set_title("(a) Held-out audit validation")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", frameon=False)
    ax.text(
        0.97,
        0.04,
        r"$\rho=.855,\ p<.001$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    ax = axes[1]
    for score in cross["model_scores"]:
        ax.scatter(
            score["feed_score"],
            score["retrieval_score"],
            s=48,
            color="#6A3D9A",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.annotate(
            MODEL_LABELS[score["model"]],
            (score["feed_score"], score["retrieval_score"]),
            xytext=CROSS_LABEL_OFFSETS[score["model"]],
            textcoords="offset points",
            fontsize=6.8,
        )
    identity_axis(ax, -1.1, 0.05)
    ax.set_xlabel("Interactive-feed susceptibility")
    ax.set_ylabel("Static-RAG susceptibility")
    ax.set_title("(b) Cross-interface transfer")
    ax.text(
        0.97,
        0.04,
        r"$\rho=.750,\ p=.033$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    ax = axes[2]
    frontier_colors = {
        "gpt-5.6-sol": "#0072B2",
        "gpt-5.6-terra": "#E69F00",
        "gpt-5.6-luna": "#CC79A7",
    }
    for cell in frontier["cells"]:
        model = cell["model"]
        ax.scatter(
            cell["audit_effect"],
            cell["full_effect"],
            s=44,
            marker=TASK_MARKERS[cell["task"]],
            color=frontier_colors[model],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            label=MODEL_LABELS[model],
        )
    identity_axis(ax, -0.85, 0.1)
    ax.set_xlabel("Five-document audit effect")
    ax.set_ylabel("45-document dossier effect")
    ax.set_title("(c) Frontier-agent boundary")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", frameon=False)
    ax.text(
        0.97,
        0.04,
        "0/18 full effects Holm-significant",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "paper_fig8_audit_validation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "paper_fig8_audit_validation.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
