"""HellaSwag benchmark evaluation — sentence completion accuracy.

Uses log-likelihood scoring to rank continuations for each example.
Evaluation is performed via an OpenAI-compatible completions endpoint
(vLLM, NIM, etc.) with ``logprobs=True``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _build_context(row: dict) -> str:
    """Build the activity context from ctx_a + ctx_b."""
    ctx_a: str = row.get("ctx_a", "").strip()
    ctx_b: str = row.get("ctx_b", "").strip()
    if ctx_b:
        return f"{ctx_a} {ctx_b}".strip()
    return ctx_a.strip()


def _normalize_text(text: str) -> str:
    """Apply HellaSwag text normalisation (remove bracketed tokens)."""
    import re
    text = re.sub(r"\[.*?\]", "", text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Log-likelihood scoring
# ---------------------------------------------------------------------------


def _get_continuation_logprob(
    client: OpenAI,
    model: str,
    context: str,
    continuation: str,
) -> tuple[float, int]:
    """Compute sum of token log-probs for ``continuation`` conditioned on ``context``.

    Returns
    -------
    (sum_logprob, n_tokens)
        sum_logprob: sum of token log-probabilities for the continuation tokens.
        n_tokens: number of continuation tokens scored.
    """
    full_text = context + continuation
    try:
        response = client.completions.create(
            model=model,
            prompt=full_text,
            max_tokens=0,  # score only; do not generate
            echo=True,     # return logprobs for the prompt itself
            logprobs=1,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("logprobs request failed: %s", exc)
        return (-1e9, 1)

    token_logprobs: list[float | None] = response.choices[0].logprobs.token_logprobs or []
    tokens: list[str] = response.choices[0].logprobs.tokens or []

    # Reconstruct text character positions to find where continuation starts
    context_len = len(context)
    cumulative = 0
    continuation_start_token_idx: int = len(tokens)  # fallback: all tokens

    for idx, tok in enumerate(tokens):
        if cumulative >= context_len:
            continuation_start_token_idx = idx
            break
        cumulative += len(tok)

    cont_logprobs = [
        lp for lp in token_logprobs[continuation_start_token_idx:]
        if lp is not None
    ]

    if not cont_logprobs:
        return (-1e9, 1)

    return (sum(cont_logprobs), len(cont_logprobs))


def _score_example(
    client: OpenAI,
    model: str,
    row: dict,
) -> dict[str, Any]:
    """Score a single HellaSwag example.

    Returns a dict with:
    - label: ground truth index (int)
    - predicted: model-predicted index based on normalised log-prob
    - correct: whether prediction matches label
    - scores: list of (sum_logprob, n_tokens) per ending
    """
    context = _build_context(row)
    label = int(row["label"])
    endings: list[str] = row["endings"]

    scores: list[tuple[float, int]] = []
    for ending in endings:
        ending = _normalize_text(ending)
        sum_lp, n_tok = _get_continuation_logprob(client, model, context, " " + ending)
        scores.append((sum_lp, n_tok))

    # acc_norm: normalise by continuation length
    norm_scores = [
        (sum_lp / max(n_tok, 1)) for sum_lp, n_tok in scores
    ]
    predicted = int(max(range(len(norm_scores)), key=lambda i: norm_scores[i]))

    return {
        "label": label,
        "predicted": predicted,
        "correct": predicted == label,
        "raw_scores": [(float(s), n) for s, n in scores],
        "norm_scores": [float(s) for s in norm_scores],
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_hellaswag_eval(
    model: str,
    endpoint: str,
    n_samples: int = 1000,
    output_json: str | None = None,
) -> dict[str, Any]:
    """Run HellaSwag evaluation.

    Parameters
    ----------
    model:
        Model name served by the endpoint.
    endpoint:
        Base URL of the OpenAI-compatible API server.
    n_samples:
        Number of validation examples to evaluate. Use -1 for the full set.
    output_json:
        Optional path to write results JSON. If None, auto-generated.

    Returns
    -------
    dict with keys: model, endpoint, n_samples, acc_norm, timestamp, per_example
    """
    client = OpenAI(base_url=endpoint, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))

    logger.info("Loading HellaSwag validation split...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")

    if n_samples != -1:
        total = min(n_samples, len(dataset))
        dataset = dataset.select(range(total))
    else:
        total = len(dataset)

    logger.info("Evaluating %d HellaSwag examples with model: %s", total, model)

    per_example: list[dict[str, Any]] = []
    n_correct = 0

    for i, row in enumerate(dataset):
        if (i + 1) % 100 == 0:
            logger.info(
                "Progress: %d/%d | acc_norm so far: %.4f",
                i + 1,
                total,
                n_correct / (i + 1),
            )

        result = _score_example(client, model, row)
        per_example.append(result)
        if result["correct"]:
            n_correct += 1

    acc_norm = n_correct / total if total > 0 else None
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    results: dict[str, Any] = {
        "model": model,
        "endpoint": endpoint,
        "n_samples": total,
        "n_correct": n_correct,
        "acc_norm": acc_norm,
        "timestamp": timestamp,
        "per_example": per_example,
    }

    # Save results
    if output_json is None:
        results_dir = Path("evals/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_").replace(":", "_")
        output_json = str(results_dir / f"hellaswag_{safe_model}_{timestamp}.json")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)

    logger.info("HellaSwag results saved to: %s", output_json)
    print(f"\n[HellaSwag] acc_norm: {acc_norm:.4f} ({n_correct}/{total})" if acc_norm is not None else "\n[HellaSwag] Evaluation complete.")  # noqa: E501

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HellaSwag sentence-completion benchmark via OpenAI-compatible endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Model name to evaluate.")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("VLLM_ENDPOINT") or os.getenv("NIM_BASE_URL", "http://localhost:8000/v1"),
        help="Base URL of the OpenAI-compatible API server.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of validation examples. Use -1 for full validation set.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write results JSON file. Auto-generated if not set.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_hellaswag_eval(
        model=args.model,
        endpoint=args.endpoint,
        n_samples=args.n_samples,
        output_json=args.output_json,
    )
