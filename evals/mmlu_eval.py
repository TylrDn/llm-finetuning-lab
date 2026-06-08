"""MMLU (Massive Multitask Language Understanding) benchmark evaluation.

Runs 5-shot MMLU evaluation against any OpenAI-compatible endpoint
(vLLM, NIM, etc.) and saves per-subject, per-group, and overall accuracy.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
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
# Subject definitions
# ---------------------------------------------------------------------------

SUBJECT_GROUPS: dict[str, list[str]] = {
    "STEM": [
        "high_school_physics",
        "high_school_mathematics",
        "high_school_chemistry",
        "high_school_computer_science",
        "college_physics",
        "college_mathematics",
        "college_chemistry",
        "college_computer_science",
        "abstract_algebra",
        "electrical_engineering",
    ],
    "Humanities": [
        "high_school_world_history",
        "high_school_us_history",
        "high_school_european_history",
        "philosophy",
        "international_law",
        "jurisprudence",
        "professional_law",
        "prehistory",
        "world_religions",
        "moral_disputes",
    ],
    "Social Sciences": [
        "high_school_economics",
        "econometrics",
        "high_school_geography",
        "high_school_psychology",
        "professional_psychology",
        "sociology",
        "us_foreign_policy",
        "human_sexuality",
        "public_relations",
        "security_studies",
    ],
    "Professional": [
        "professional_medicine",
        "clinical_knowledge",
        "medical_genetics",
        "anatomy",
        "college_medicine",
        "nutrition",
        "professional_accounting",
        "business_ethics",
        "management",
        "marketing",
    ],
}

# Flat list of all subjects
ALL_SUBJECTS: list[str] = [s for subjects in SUBJECT_GROUPS.values() for s in subjects]

CHOICES = ["A", "B", "C", "D"]


def _subject_to_group(subject: str) -> str:
    for group, subjects in SUBJECT_GROUPS.items():
        if subject in subjects:
            return group
    return "Other"


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_example(row: dict, include_answer: bool = True) -> str:
    """Format a single MMLU example as 'Question: ...\nA. ...\nAnswer: X'."""
    question = row["question"].strip()
    choices = row["choices"]  # list of 4 strings
    prompt = f"Question: {question}\n"
    for i, choice_text in enumerate(choices):
        prompt += f"{CHOICES[i]}. {choice_text}\n"
    prompt += "Answer:"
    if include_answer:
        answer_letter = CHOICES[row["answer"]]
        prompt += f" {answer_letter}\n"
    return prompt


def _build_few_shot_prompt(
    dev_examples: list[dict],
    test_row: dict,
    n_shots: int,
) -> str:
    """Build a few-shot prompt for a given test question."""
    header = (
        "The following are multiple choice questions (with answers) about "
        f"{test_row.get('subject', 'a topic')}.\n\n"
    )
    shots = dev_examples[:n_shots]
    few_shot_block = "".join(_format_example(ex, include_answer=True) for ex in shots)
    question_block = _format_example(test_row, include_answer=False)
    return header + few_shot_block + question_block


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


def _extract_answer_letter(response_text: str) -> str | None:
    """Extract the first A/B/C/D answer letter from a model response."""
    text = response_text.strip()
    # Try direct match at start
    m = re.match(r"^([A-D])\b", text)
    if m:
        return m.group(1)
    # Try "Answer: X" pattern
    m = re.search(r"Answer:\s*([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Last resort: first standalone letter
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)
    return None


def _query_model(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 16,
    temperature: float = 0.0,
) -> str:
    """Send a completion request and return the text response."""
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n", "Question:"],
    )
    return response.choices[0].text


# ---------------------------------------------------------------------------
# Subject evaluation
# ---------------------------------------------------------------------------


def evaluate_subject(
    client: OpenAI,
    model: str,
    subject: str,
    n_shots: int = 5,
) -> dict[str, Any]:
    """Evaluate a single MMLU subject.

    Returns
    -------
    dict with keys: subject, n_correct, n_total, accuracy
    """
    logger.info("Evaluating subject: %s", subject)

    try:
        dev_data = load_dataset("cais/mmlu", subject, split="dev")
        test_data = load_dataset("cais/mmlu", subject, split="test")
    except Exception as exc:
        logger.warning("Failed to load subject '%s': %s", subject, exc)
        return {"subject": subject, "n_correct": 0, "n_total": 0, "accuracy": None, "error": str(exc)}

    dev_examples = [
        {
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "subject": subject,
        }
        for row in dev_data
    ]
    test_examples = [
        {
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "subject": subject,
        }
        for row in test_data
    ]

    n_correct = 0
    n_total = len(test_examples)

    for i, test_row in enumerate(test_examples):
        prompt = _build_few_shot_prompt(dev_examples, test_row, n_shots)
        try:
            response_text = _query_model(client, model, prompt)
            predicted = _extract_answer_letter(response_text)
            correct_letter = CHOICES[test_row["answer"]]
            if predicted == correct_letter:
                n_correct += 1
        except Exception as exc:
            logger.warning("Inference error on subject=%s example=%d: %s", subject, i, exc)

    accuracy = n_correct / n_total if n_total > 0 else None
    logger.info(
        "Subject %s: %d/%d correct (acc=%.4f)",
        subject, n_correct, n_total, accuracy or 0.0,
    )
    return {
        "subject": subject,
        "group": _subject_to_group(subject),
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def run_mmlu_eval(
    model: str,
    endpoint: str,
    subjects: list[str],
    n_shots: int = 5,
    output_json: str | None = None,
) -> dict[str, Any]:
    """Run MMLU evaluation across specified subjects.

    Parameters
    ----------
    model:
        Model name served by the endpoint.
    endpoint:
        Base URL of the OpenAI-compatible API server.
    subjects:
        List of MMLU subject names; pass ALL_SUBJECTS for full benchmark.
    n_shots:
        Number of few-shot examples per question (default 5).
    output_json:
        Optional path to write results JSON. If None, auto-generated.

    Returns
    -------
    dict with keys: overall, by_group, by_subject, model, timestamp
    """
    client = OpenAI(base_url=endpoint, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))

    by_subject: dict[str, Any] = {}
    for subject in subjects:
        result = evaluate_subject(client, model, subject, n_shots=n_shots)
        by_subject[subject] = result

    # Compute per-group accuracy
    by_group: dict[str, dict[str, Any]] = {}
    for group in SUBJECT_GROUPS:
        group_subjects = [s for s in subjects if _subject_to_group(s) == group]
        if not group_subjects:
            continue
        group_correct = sum(
            by_subject[s]["n_correct"] for s in group_subjects if by_subject[s].get("accuracy") is not None
        )
        group_total = sum(
            by_subject[s]["n_total"] for s in group_subjects if by_subject[s].get("accuracy") is not None
        )
        by_group[group] = {
            "n_correct": group_correct,
            "n_total": group_total,
            "accuracy": group_correct / group_total if group_total > 0 else None,
        }

    # Compute overall accuracy
    total_correct = sum(
        v["n_correct"] for v in by_subject.values() if v.get("accuracy") is not None
    )
    total_examples = sum(
        v["n_total"] for v in by_subject.values() if v.get("accuracy") is not None
    )
    overall_accuracy = total_correct / total_examples if total_examples > 0 else None

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results: dict[str, Any] = {
        "model": model,
        "endpoint": endpoint,
        "n_shots": n_shots,
        "timestamp": timestamp,
        "overall": overall_accuracy,
        "by_group": by_group,
        "by_subject": by_subject,
    }

    # Save results
    if output_json is None:
        results_dir = Path("evals/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_").replace(":", "_")
        output_json = str(results_dir / f"mmlu_{safe_model}_{timestamp}.json")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)

    logger.info("MMLU results saved to: %s", output_json)
    print(f"\n[MMLU] Overall accuracy: {overall_accuracy:.4f}" if overall_accuracy else "\n[MMLU] Evaluation complete.")
    for group, stats in by_group.items():
        acc = stats.get("accuracy")
        print(f"  {group}: {acc:.4f}" if acc is not None else f"  {group}: N/A")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MMLU benchmark evaluation via OpenAI-compatible endpoint.",
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
        "--subjects",
        nargs="+",
        default=["all"],
        help="Subjects to evaluate; use 'all' for the full benchmark.",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=5,
        help="Number of few-shot examples per question.",
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
    subjects = ALL_SUBJECTS if args.subjects == ["all"] else args.subjects
    run_mmlu_eval(
        model=args.model,
        endpoint=args.endpoint,
        subjects=subjects,
        n_shots=args.n_shots,
        output_json=args.output_json,
    )
