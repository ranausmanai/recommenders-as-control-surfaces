"""04_jima_compare: Ji Ma-style static stance vectors vs empirical drift PC1.

For each topic, build a 'static stance vector' = mean(act("I strongly support X"))
minus mean(act("I strongly oppose X")) at the best layer. Then compare it to the
empirical drift PC1 from notebook 02 via cosine similarity.

Hypothesis: cos similarity is NOT close to 1.0 — the dimension along which the
agent's beliefs DRIFT under feed exposure is different from the static-prompt
stance direction. That difference is the novelty over Ji Ma 2026.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
fp = import_module("01_fingerprint")
from activation_hooks import load_model, ActivationCapture

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"

CONTRAST_PAIRS = {
    "remote_work": [
        ("I strongly believe remote work is the future of professional life.",
         "I strongly believe everyone should return to the office full-time."),
        ("Working from home is clearly better for productivity and well-being.",
         "In-office work is clearly better for productivity and collaboration."),
        ("Remote work has proven its value beyond any doubt.",
         "Return-to-office mandates are clearly the right call."),
    ],
    "ai_regulation": [
        ("I strongly support keeping AI development open-source and unregulated.",
         "I strongly support strict government regulation of AI development."),
        ("Open-source AI models must remain freely available to everyone.",
         "Open-source AI models pose unacceptable risks and must be restricted."),
        ("AI regulation will only stifle innovation and concentrate power.",
         "AI regulation is urgently needed to prevent serious societal harm."),
    ],
    "nuclear_energy": [
        ("I strongly support nuclear energy as our best climate solution.",
         "I strongly oppose nuclear energy in any form."),
        ("Nuclear power is clean, safe, and essential for decarbonization.",
         "Nuclear power is dangerous, expensive, and a distraction from renewables."),
        ("We urgently need more nuclear reactors to fight climate change.",
         "We must shut down all nuclear plants — they are too risky."),
    ],
}


def capture_text(tok, model, text: str, device="mps") -> torch.Tensor:
    """Return (n_layers, hidden) fp16 activation at final token of `text`."""
    enc = tok(text, return_tensors="pt").to(device)
    with ActivationCapture(model) as cap:
        with torch.no_grad():
            _ = model(**enc)
        return cap.last(seq_index=-1)


def build_static_stance_vector(tok, model, pairs: list[tuple[str, str]], layer: int, device="mps") -> np.ndarray:
    """Mean of (support - oppose) at given layer."""
    diffs = []
    for support, oppose in pairs:
        a_sup = capture_text(tok, model, support, device).numpy().astype(np.float32)[layer]
        a_opp = capture_text(tok, model, oppose, device).numpy().astype(np.float32)[layer]
        diffs.append(a_sup - a_opp)
    return np.mean(diffs, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen_Qwen3.5-0.8B"
    model_id = model_name.replace("_", "/", 1)
    fp_summary = json.loads((ROOT / "results" / "01_fingerprint.json").read_text())
    L = int(fp_summary["best_layer"])

    print(f"Loading {model_id} for stance vector extraction ...")
    tok, model = load_model(model_id)

    # Compute empirical drift PC1 at L per topic (within-topic PCA on the trajectory)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X, y, meta, conditions = fp.load_dataset(model_name)

    results_by_topic = {}
    for topic in CONTRAST_PAIRS:
        idx = [i for i, m in enumerate(meta) if m["topic"] == topic]
        if not idx:
            print(f"  skipping {topic}: no activations")
            continue
        Xi = X[idx][:, L, :]
        sc = StandardScaler().fit(Xi)
        Z = sc.transform(Xi)
        pca = PCA(n_components=1).fit(Z)
        # Map PC1 back to original space: pca.components_ is in standardized space.
        # For directional comparison we should use the principal axis in the original
        # (centered) space: scale-back by sc.scale_.
        pc1_std = pca.components_[0]
        # Approximate axis in original space: divide by per-feature std.
        pc1_orig = pc1_std / (sc.scale_ + 1e-9)

        stance_vec = build_static_stance_vector(tok, model, CONTRAST_PAIRS[topic], layer=L)
        cs = cosine(pc1_orig, stance_vec)
        # Also report sign-invariant similarity (since PCA sign is arbitrary)
        cs_abs = abs(cs)

        print(f"  topic={topic}: cosine(PC1, stance_vec) = {cs:+.3f}  (|.|={cs_abs:.3f})")
        results_by_topic[topic] = {
            "cosine": cs,
            "cosine_abs": cs_abs,
            "n_samples": len(idx),
        }

    # Plot bar of |cosine| per topic
    if results_by_topic:
        topics = list(results_by_topic.keys())
        vals = [results_by_topic[t]["cosine_abs"] for t in topics]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(topics, vals, color="steelblue")
        ax.axhline(1.0, ls="--", color="grey", label="perfect alignment")
        ax.set_ylabel("|cos(empirical drift PC1, static stance vec)|")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Drift direction vs static stance vector @ L{L}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "04_jima_alignment.png", dpi=140)
        print(f"Saved {FIGS / '04_jima_alignment.png'}")

    out = {"best_layer": L, "per_topic": results_by_topic}
    with open(ROOT / "results" / "04_jima_compare.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
