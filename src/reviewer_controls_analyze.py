"""Prespecified statistical analysis for reviewer-control experiments."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from feed_policies import load_pool
from reviewer_controls import INTENSITIES, MATCHED_POOL


POSITION = {"A": 0, "B": 1, "C": 2}
PRIMARY_COMPARISONS = {
    "matched_selection": [("matched_rto", "matched_remote")],
    "pure_order": [("order_rto_last", "order_remote_last")],
}


def load_records(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def paired_rows(
    records: Iterable[dict],
    model: str,
    left: str,
    right: str,
    *,
    topic: str = "remote_work",
) -> list[tuple[int, int]]:
    by_condition: dict[str, dict[int, int]] = defaultdict(dict)
    for record in records:
        if (
            record["model"] != model
            or record["topic"] != topic
            or record.get("choice") not in POSITION
        ):
            continue
        by_condition[record["condition"]][int(record["seed"])] = POSITION[record["choice"]]
    seeds = sorted(set(by_condition[left]) & set(by_condition[right]))
    return [(by_condition[left][seed], by_condition[right][seed]) for seed in seeds]


def paired_randomization_pvalue(
    differences: np.ndarray, *, seed: int = 260600914, draws: int = 100000
) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(np.mean(nonzero)))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1, 1), size=(draws, len(nonzero)))
    permuted = np.abs(np.mean(signs * nonzero, axis=1))
    return float((np.sum(permuted >= observed) + 1) / (draws + 1))


def bootstrap_mean_ci(
    differences: np.ndarray, *, seed: int = 260600914, draws: int = 10000
) -> tuple[float, float]:
    if len(differences) == 0:
        return math.nan, math.nan
    rng = random.Random(seed)
    means = []
    values = list(map(float, differences))
    for _ in range(draws):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def summarize_condition(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("choice") in POSITION]
    counts = Counter(row["choice"] for row in valid)
    n = len(valid)
    proportions = {}
    for choice in ("A", "B", "C"):
        lo, hi = wilson(counts[choice], n)
        proportions[choice] = {
            "count": counts[choice],
            "proportion": counts[choice] / n if n else math.nan,
            "ci95": [lo, hi],
        }
    positions = [POSITION[row["choice"]] for row in valid]
    return {
        "n_valid": n,
        "n_total": len(rows),
        "n_unparsable": len(rows) - n,
        "choices": proportions,
        "mean_position": float(np.mean(positions)) if positions else math.nan,
    }


def holm_adjust(comparisons: list[dict]) -> None:
    if not comparisons:
        return
    adjusted = multipletests(
        [comparison["p_raw"] for comparison in comparisons],
        alpha=0.05,
        method="holm",
    )
    for comparison, reject, p_adjusted in zip(comparisons, adjusted[0], adjusted[1]):
        comparison["p_holm"] = float(p_adjusted)
        comparison["reject_holm_005"] = bool(reject)


def comparison_record(
    records: list[dict],
    *,
    experiment: str,
    model: str,
    left: str,
    right: str,
) -> dict | None:
    pairs_data = paired_rows(records, model, left, right)
    if not pairs_data:
        return None
    diffs = np.array(
        [left_value - right_value for left_value, right_value in pairs_data]
    )
    lo, hi = bootstrap_mean_ci(diffs)
    return {
        "experiment": experiment,
        "model": model,
        "left": left,
        "right": right,
        "n_pairs": len(pairs_data),
        "mean_ordinal_difference_left_minus_right": float(np.mean(diffs)),
        "mean_difference_ci95": [lo, hi],
        "p_raw": paired_randomization_pvalue(diffs),
    }


def audit_matched_pool() -> dict:
    pool = load_pool(MATCHED_POOL, topic="remote_work")
    by_stance = {}
    for stance in (-2, -1, 0, 1, 2):
        rows = [post for post in pool if post["stance"] == stance]
        lengths = [len(post["text"].split()) for post in rows]
        by_stance[str(stance)] = {
            "n": len(rows),
            "intensity_counts": dict(
                sorted(Counter(post["intensity"] for post in rows).items())
            ),
            "word_count_mean": float(np.mean(lengths)),
            "word_count_min": min(lengths),
            "word_count_max": max(lengths),
        }
    return {
        "path": str(MATCHED_POOL),
        "n": len(pool),
        "all_topic_remote_work": all(
            post.get("topic") == "remote_work" for post in pool
        ),
        "all_single_generator": len(
            {post.get("generator") for post in pool}
        ) == 1,
        "generators": sorted(
            str(generator) for generator in {post.get("generator") for post in pool}
        ),
        "contains_adversarial_flag": any(post.get("adversarial") for post in pool),
        "by_stance": by_stance,
        "required_intensities": list(INTENSITIES),
    }


def analyze(records: list[dict]) -> dict:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[
            (record["model"], record.get("experiment", "unknown"), record["condition"])
        ].append(record)

    summaries = {}
    for (model, experiment, condition), rows in sorted(grouped.items()):
        by_topic: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_topic[row["topic"]].append(row)
        for topic, topic_rows in sorted(by_topic.items()):
            summaries[f"{model}|{topic}|{experiment}|{condition}"] = (
                summarize_condition(topic_rows)
            )

    defaults = {}
    for (model, experiment, condition), rows in sorted(grouped.items()):
        if experiment != "no_feed" or condition != "no_feed":
            continue
        by_topic: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_topic[row["topic"]].append(row)
        for topic, topic_rows in sorted(by_topic.items()):
            summary = summarize_condition(topic_rows)
            counts = {
                choice: summary["choices"][choice]["count"] for choice in ("A", "B", "C")
            }
            modal = max(counts, key=counts.get) if sum(counts.values()) else None
            modal_share = (
                counts[modal] / sum(counts.values()) if modal is not None else math.nan
            )
            defaults[f"{model}|{topic}"] = {
                **summary,
                "modal_choice": modal,
                "modal_share": modal_share,
                "non_ceiling_for_asymmetry": bool(modal is not None and modal_share < 0.95),
            }

    comparisons = []
    models = sorted({record["model"] for record in records})
    for experiment, pairs in PRIMARY_COMPARISONS.items():
        family = []
        for model in models:
            available = {
                record["condition"]
                for record in records
                if record["model"] == model and record.get("experiment") == experiment
            }
            for left, right in pairs:
                if left not in available or right not in available:
                    continue
                comparison = comparison_record(
                    records,
                    experiment=experiment,
                    model=model,
                    left=left,
                    right=right,
                )
                if comparison is not None:
                    family.append(comparison)
        holm_adjust(family)
        comparisons.extend(family)

    secondary_specs = {
        "matched_selection": [
            ("matched_rto", "matched_balanced"),
            ("matched_remote", "matched_balanced"),
            ("matched_rto", "no_feed"),
            ("matched_remote", "no_feed"),
        ],
        "pure_order": [
            ("order_rto_last", "order_interleaved"),
            ("order_remote_last", "order_interleaved"),
        ],
    }
    secondary = []
    for experiment, pairs in secondary_specs.items():
        family = []
        for model in models:
            available = {
                record["condition"]
                for record in records
                if record["model"] == model
            }
            for left, right in pairs:
                if left not in available or right not in available:
                    continue
                comparison = comparison_record(
                    records,
                    experiment=experiment,
                    model=model,
                    left=left,
                    right=right,
                )
                if comparison is not None:
                    family.append(comparison)
        holm_adjust(family)
        secondary.extend(family)

    asymmetry = []
    for model in models:
        default = defaults.get(f"{model}|remote_work")
        if not default:
            continue
        needed = {"no_feed", "matched_rto", "matched_remote"}
        available = {
            record["condition"] for record in records if record["model"] == model
        }
        if not needed.issubset(available):
            continue
        by_condition: dict[str, dict[int, int]] = defaultdict(dict)
        for record in records:
            if (
                record["model"] != model
                or record["topic"] != "remote_work"
                or record.get("choice") not in POSITION
            ):
                continue
            by_condition[record["condition"]][int(record["seed"])] = POSITION[
                record["choice"]
            ]
        seeds = sorted(
            set(by_condition["no_feed"])
            & set(by_condition["matched_rto"])
            & set(by_condition["matched_remote"])
        )
        toward_rto = np.array(
            [
                by_condition["no_feed"][seed] - by_condition["matched_rto"][seed]
                for seed in seeds
            ]
        )
        toward_remote = np.array(
            [
                by_condition["matched_remote"][seed] - by_condition["no_feed"][seed]
                for seed in seeds
            ]
        )
        difference = toward_rto - toward_remote
        lo, hi = bootstrap_mean_ci(difference)
        asymmetry.append(
            {
                "model": model,
                "n_triplets": len(seeds),
                "feed_free_modal_choice": default["modal_choice"],
                "feed_free_modal_share": default["modal_share"],
                "eligible_non_ceiling": default["non_ceiling_for_asymmetry"],
                "mean_shift_toward_rto": float(np.mean(toward_rto)),
                "mean_shift_toward_remote": float(np.mean(toward_remote)),
                "difference_rto_minus_remote": float(np.mean(difference)),
                "difference_ci95": [lo, hi],
                "p_raw": paired_randomization_pvalue(difference),
            }
        )
    holm_adjust(asymmetry)

    return {
        "analysis_version": 2,
        "n_records": len(records),
        "matched_pool_audit": audit_matched_pool(),
        "condition_summaries": summaries,
        "feed_free_defaults": defaults,
        "primary_comparisons": comparisons,
        "secondary_comparisons": secondary,
        "default_direction_asymmetry": asymmetry,
        "interpretation_guardrails": {
            "default_claim_requires_direct_no_feed_measurement": True,
            "asymmetry_requires_modal_share_below": 0.95,
            "dose_claim": "dose-dependent trend; no precise threshold claim",
        },
    }


def markdown_report(result: dict) -> str:
    lines = [
        "# Reviewer-Control Results",
        "",
        f"Records analyzed: {result['n_records']}",
        "",
        "## Feed-Free Defaults",
        "",
        "| Model | Topic | A/B/C | Modal | Share | Non-ceiling? |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, row in sorted(result["feed_free_defaults"].items()):
        model, topic = key.split("|", 1)
        counts = "/".join(str(row["choices"][choice]["count"]) for choice in ("A", "B", "C"))
        lines.append(
            f"| {model} | {topic} | {counts} | {row['modal_choice']} | "
            f"{row['modal_share']:.2f} | {'yes' if row['non_ceiling_for_asymmetry'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Primary Comparisons",
            "",
            "| Experiment | Model | Comparison | n | Mean ordinal diff | 95% CI | Raw p | Holm p |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["primary_comparisons"]:
        lo, hi = row["mean_difference_ci95"]
        lines.append(
            f"| {row['experiment']} | {row['model']} | "
            f"{row['left']} - {row['right']} | {row['n_pairs']} | "
            f"{row['mean_ordinal_difference_left_minus_right']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {row['p_raw']:.4g} | {row['p_holm']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Default-Direction Asymmetry",
            "",
            "| Model | n | Default/share | Non-ceiling? | RTO shift | Remote shift | Difference | Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["default_direction_asymmetry"]:
        lines.append(
            f"| {row['model']} | {row['n_triplets']} | "
            f"{row['feed_free_modal_choice']}/{row['feed_free_modal_share']:.2f} | "
            f"{'yes' if row['eligible_non_ceiling'] else 'no'} | "
            f"{row['mean_shift_toward_rto']:.3f} | "
            f"{row['mean_shift_toward_remote']:.3f} | "
            f"{row['difference_rto_minus_remote']:.3f} | {row['p_holm']:.4g} |"
        )
    lines.extend(
        [
            "",
            "Dose-response wording is restricted to **dose-dependent trend**. "
            "These controls do not test or claim a precise onset threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("results/reviewer_controls.jsonl")
    )
    parser.add_argument(
        "--json-out", type=Path, default=Path("results/reviewer_controls_analysis.json")
    )
    parser.add_argument(
        "--md-out", type=Path, default=Path("results/reviewer_controls_report.md")
    )
    args = parser.parse_args()

    records = load_records(args.input)
    result = analyze(records)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    args.md_out.write_text(markdown_report(result))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
