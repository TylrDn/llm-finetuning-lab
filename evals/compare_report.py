"""Generate comparison report: base model vs fine-tuned across MMLU and HellaSwag.

Loads the most-recent MMLU and HellaSwag JSON result files from a base model
results directory and a fine-tuned model results directory, then produces:
  1. A markdown comparison report with delta columns.
  2. A W&B Artifact upload (if WANDB_API_KEY is set).
  3. A summary table printed to stdout.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Unicode colour indicators
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"
UP_ARROW = "▲"
DOWN_ARROW = "▼"
DASH = "—"


# ---------------------------------------------------------------------------
# Result loading helpers
# ---------------------------------------------------------------------------


def _find_latest_file(directory: str, prefix: str) -> str | None:
    """Find the most recently modified JSON file matching ``prefix`` in ``directory``."""
    pattern = str(Path(directory) / f"{prefix}*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Failed to load result file %s: %s", path, exc)
        return None


def _load_results(results_dir: str) -> dict[str, dict[str, Any] | None]:
    """Load latest MMLU and HellaSwag results from a directory."""
    mmlu_path = _find_latest_file(results_dir, "mmlu_")
    hellaswag_path = _find_latest_file(results_dir, "hellaswag_")

    logger.info("Loading MMLU results from: %s", mmlu_path)
    logger.info("Loading HellaSwag results from: %s", hellaswag_path)

    return {
        "mmlu": _load_json(mmlu_path),
        "hellaswag": _load_json(hellaswag_path),
        "mmlu_path": mmlu_path,
        "hellaswag_path": hellaswag_path,
    }


# ---------------------------------------------------------------------------
# Delta formatting
# ---------------------------------------------------------------------------


def _fmt_acc(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _fmt_delta(base: float | None, finetuned: float | None) -> str:
    """Format a delta value with direction arrow and sign."""
    if base is None or finetuned is None:
        return DASH
    delta = finetuned - base
    pct = delta * 100
    sign = "+" if pct >= 0 else ""
    arrow = UP_ARROW if pct > 0 else (DOWN_ARROW if pct < 0 else DASH)
    return f"{arrow} {sign}{pct:.2f}%"


def _delta_color(base: float | None, finetuned: float | None) -> str:
    """Return unicode coloured delta for terminal output."""
    if base is None or finetuned is None:
        return DASH
    delta = finetuned - base
    pct = delta * 100
    sign = "+" if pct >= 0 else ""
    color = _GREEN if pct > 0 else (_RED if pct < 0 else "")
    arrow = UP_ARROW if pct > 0 else (DOWN_ARROW if pct < 0 else DASH)
    reset = _RESET if color else ""
    return f"{color}{arrow} {sign}{pct:.2f}%{reset}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _build_summary_table_md(
    base_mmlu: dict | None,
    ft_mmlu: dict | None,
    base_hs: dict | None,
    ft_hs: dict | None,
) -> str:
    rows = []
    rows.append("| Benchmark | Base | Fine-tuned | Delta |")
    rows.append("|-----------|------|------------|-------|")

    # MMLU overall
    base_overall = base_mmlu.get("overall") if base_mmlu else None
    ft_overall = ft_mmlu.get("overall") if ft_mmlu else None
    rows.append(
        f"| MMLU (overall) | {_fmt_acc(base_overall)} | {_fmt_acc(ft_overall)} | {_fmt_delta(base_overall, ft_overall)} |"  # noqa: E501
    )

    # MMLU by group
    all_groups = set()
    if base_mmlu:
        all_groups.update(base_mmlu.get("by_group", {}).keys())
    if ft_mmlu:
        all_groups.update(ft_mmlu.get("by_group", {}).keys())

    for group in sorted(all_groups):
        base_acc = (
            (base_mmlu.get("by_group", {}).get(group, {}) or {}).get("accuracy")
            if base_mmlu
            else None
        )
        ft_acc = (
            (ft_mmlu.get("by_group", {}).get(group, {}) or {}).get("accuracy")
            if ft_mmlu
            else None
        )
        rows.append(
            f"| MMLU — {group} | {_fmt_acc(base_acc)} | {_fmt_acc(ft_acc)} | {_fmt_delta(base_acc, ft_acc)} |"  # noqa: E501
        )

    # HellaSwag
    base_hs_acc = base_hs.get("acc_norm") if base_hs else None
    ft_hs_acc = ft_hs.get("acc_norm") if ft_hs else None
    rows.append(
        f"| HellaSwag (acc_norm) | {_fmt_acc(base_hs_acc)} | {_fmt_acc(ft_hs_acc)} | {_fmt_delta(base_hs_acc, ft_hs_acc)} |"  # noqa: E501
    )

    return "\n".join(rows)


def _build_subject_table_md(
    base_mmlu: dict | None,
    ft_mmlu: dict | None,
) -> str:
    if not base_mmlu and not ft_mmlu:
        return "_No MMLU subject data available._"

    all_subjects: set[str] = set()
    if base_mmlu:
        all_subjects.update(base_mmlu.get("by_subject", {}).keys())
    if ft_mmlu:
        all_subjects.update(ft_mmlu.get("by_subject", {}).keys())

    rows = []
    rows.append("| Subject | Group | Base | Fine-tuned | Delta |")
    rows.append("|---------|-------|------|------------|-------|")

    for subject in sorted(all_subjects):
        base_sub = (base_mmlu.get("by_subject", {}).get(subject) or {}) if base_mmlu else {}
        ft_sub = (ft_mmlu.get("by_subject", {}).get(subject) or {}) if ft_mmlu else {}
        base_acc = base_sub.get("accuracy")
        ft_acc = ft_sub.get("accuracy")
        group = ft_sub.get("group") or base_sub.get("group") or "—"
        rows.append(
            f"| {subject} | {group} | {_fmt_acc(base_acc)} | {_fmt_acc(ft_acc)} | {_fmt_delta(base_acc, ft_acc)} |"  # noqa: E501
        )

    return "\n".join(rows)


def _interpret_results(
    base_mmlu: dict | None,
    ft_mmlu: dict | None,
    base_hs: dict | None,
    ft_hs: dict | None,
) -> str:
    lines: list[str] = []

    if base_mmlu and ft_mmlu:
        base_overall = base_mmlu.get("overall")
        ft_overall = ft_mmlu.get("overall")
        if base_overall is not None and ft_overall is not None:
            delta = (ft_overall - base_overall) * 100
            direction = "improved" if delta > 0 else "degraded"
            lines.append(
                f"Fine-tuning **{direction}** MMLU overall accuracy by "
                f"**{abs(delta):.2f}%** "
                f"({_fmt_acc(base_overall)} → {_fmt_acc(ft_overall)})."
            )

        # Per-group interpretation
        all_groups = set(base_mmlu.get("by_group", {}).keys()) | set(ft_mmlu.get("by_group", {}).keys())  # noqa: E501
        improvements: list[tuple[str, float]] = []
        regressions: list[tuple[str, float]] = []

        for group in sorted(all_groups):
            base_acc = (base_mmlu.get("by_group", {}).get(group) or {}).get("accuracy")
            ft_acc = (ft_mmlu.get("by_group", {}).get(group) or {}).get("accuracy")
            if base_acc is not None and ft_acc is not None:
                delta = (ft_acc - base_acc) * 100
                if delta > 0.1:
                    improvements.append((group, delta))
                elif delta < -0.1:
                    regressions.append((group, abs(delta)))

        if improvements:
            parts = ", ".join(f"{g} (+{d:.2f}%)" for g, d in improvements)
            lines.append(f"- **Improvements**: {parts}")
        if regressions:
            parts = ", ".join(f"{g} (-{d:.2f}%)" for g, d in regressions)
            lines.append(f"- **Regressions**: {parts}")

    if base_hs and ft_hs:
        base_acc = base_hs.get("acc_norm")
        ft_acc = ft_hs.get("acc_norm")
        if base_acc is not None and ft_acc is not None:
            delta = (ft_acc - base_acc) * 100
            direction = "improved" if delta > 0 else "degraded"
            lines.append(
                f"HellaSwag acc_norm **{direction}** by **{abs(delta):.2f}%** "
                f"({_fmt_acc(base_acc)} → {_fmt_acc(ft_acc)})."
            )

    return "\n".join(lines) if lines else "_Insufficient data for interpretation._"


def _build_markdown_report(
    base_results: dict,
    ft_results: dict,
    base_mmlu: dict | None,
    ft_mmlu: dict | None,
    base_hs: dict | None,
    ft_hs: dict | None,
    timestamp: str,
) -> str:
    base_model = (base_mmlu or base_hs or {}).get("model", "base")
    ft_model = (ft_mmlu or ft_hs or {}).get("model", "fine-tuned")

    _hs_b = base_results.get("hellaswag_path") or "N/A"
    _hs_ft = ft_results.get("hellaswag_path") or "N/A"
    report = f"""# Evaluation Comparison Report

**Generated:** {timestamp}
**Base model:** `{base_model}`
**Fine-tuned model:** `{ft_model}`

---

## Summary

{_build_summary_table_md(base_mmlu, ft_mmlu, base_hs, ft_hs)}

---

## Interpretation

{_interpret_results(base_mmlu, ft_mmlu, base_hs, ft_hs)}

---

## MMLU Per-Subject Breakdown

{_build_subject_table_md(base_mmlu, ft_mmlu)}

---

## Data Sources

| | Base | Fine-tuned |
|--|------|------------|
| MMLU | `{base_results.get("mmlu_path") or "N/A"}` | `{ft_results.get("mmlu_path") or "N/A"}` |
| HellaSwag | `{_hs_b}` | `{_hs_ft}` |
"""
    return report


# ---------------------------------------------------------------------------
# W&B artifact upload
# ---------------------------------------------------------------------------


def _upload_wandb_artifact(report_path: str, model_name: str) -> None:
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logger.info("WANDB_API_KEY not set; skipping W&B artifact upload.")
        return

    try:
        import wandb

        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "llm-finetuning-lab"),
            name=f"eval-comparison-{model_name}",
            job_type="eval-comparison",
        )
        artifact = wandb.Artifact(
            name=f"comparison-report-{model_name}",
            type="eval-report",
            description="Base vs fine-tuned model comparison report (MMLU + HellaSwag)",
        )
        artifact.add_file(report_path)
        run.log_artifact(artifact)
        run.finish()
        logger.info("W&B artifact uploaded: %s", report_path)
    except ImportError:
        logger.warning("wandb not installed; skipping artifact upload.")
    except Exception as exc:
        logger.warning("W&B artifact upload failed: %s", exc)


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------


def generate_comparison(
    base_results_dir: str,
    finetuned_results_dir: str,
    output_dir: str,
) -> None:
    """Generate base vs fine-tuned comparison report.

    Parameters
    ----------
    base_results_dir:
        Directory containing base model MMLU/HellaSwag JSON result files.
    finetuned_results_dir:
        Directory containing fine-tuned model result files.
    output_dir:
        Directory where the markdown report will be saved.
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load results
    base_results = _load_results(base_results_dir)
    ft_results = _load_results(finetuned_results_dir)

    base_mmlu: dict | None = base_results["mmlu"]
    ft_mmlu: dict | None = ft_results["mmlu"]
    base_hs: dict | None = base_results["hellaswag"]
    ft_hs: dict | None = ft_results["hellaswag"]

    if not any([base_mmlu, ft_mmlu, base_hs, ft_hs]):
        logger.error("No result files found in either directory. Cannot generate report.")
        return

    # Build report
    report_md = _build_markdown_report(
        base_results, ft_results, base_mmlu, ft_mmlu, base_hs, ft_hs, timestamp
    )

    # Save report
    ts_safe = timestamp.replace(":", "").replace("-", "")
    report_filename = f"comparison_report_{ts_safe}.md"
    report_path = output_dir_path / report_filename
    with open(report_path, "w") as fh:
        fh.write(report_md)

    logger.info("Comparison report saved to: %s", report_path)

    # Print summary to stdout
    ft_model = (ft_mmlu or ft_hs or {}).get("model", "fine-tuned")
    base_model = (base_mmlu or base_hs or {}).get("model", "base")

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPARISON: {base_model} vs {ft_model}")
    print(f"{'='*60}")
    print(f"{'Benchmark':<30} {'Base':>10} {'Fine-tuned':>12} {'Delta':>14}")
    print(f"{'-'*68}")

    def _print_row(label: str, base_acc: float | None, ft_acc: float | None) -> None:
        print(
            f"{label:<30} {_fmt_acc(base_acc):>10} {_fmt_acc(ft_acc):>12} {_delta_color(base_acc, ft_acc):>14}"  # noqa: E501
        )

    if base_mmlu or ft_mmlu:
        _print_row(
            "MMLU (overall)",
            (base_mmlu or {}).get("overall"),
            (ft_mmlu or {}).get("overall"),
        )
        all_groups = set()
        if base_mmlu:
            all_groups.update(base_mmlu.get("by_group", {}).keys())
        if ft_mmlu:
            all_groups.update(ft_mmlu.get("by_group", {}).keys())
        for group in sorted(all_groups):
            _print_row(
                f"  MMLU — {group}",
                ((base_mmlu or {}).get("by_group", {}).get(group) or {}).get("accuracy"),
                ((ft_mmlu or {}).get("by_group", {}).get(group) or {}).get("accuracy"),
            )

    if base_hs or ft_hs:
        _print_row(
            "HellaSwag (acc_norm)",
            (base_hs or {}).get("acc_norm"),
            (ft_hs or {}).get("acc_norm"),
        )

    print(f"{'='*60}")
    print(f"Report saved to: {report_path}")

    # W&B artifact upload
    _upload_wandb_artifact(str(report_path), ft_model.replace("/", "_"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate base vs fine-tuned model comparison report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Directory containing base model evaluation results (JSON files).",
    )
    parser.add_argument(
        "--finetuned-dir",
        type=str,
        required=True,
        help="Directory containing fine-tuned model evaluation results (JSON files).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evals/results",
        help="Directory to write the markdown comparison report.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    generate_comparison(
        base_results_dir=args.base_dir,
        finetuned_results_dir=args.finetuned_dir,
        output_dir=args.output_dir,
    )
