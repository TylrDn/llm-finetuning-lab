"""Post-training evaluation with LangSmith."""
import os

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from openai import OpenAI

load_dotenv()

ls_client = Client()


def exact_match_evaluator(run, example):
    pred = run.outputs.get("response", "").strip().lower()
    expected = example.outputs.get("expected", "").strip().lower()
    return {"key": "exact_match", "score": int(pred == expected)}


def run_eval(dataset_name: str = "finetuning-eval-v1", model_endpoint: str = None):
    client = OpenAI(
        base_url=model_endpoint or os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1"),
        api_key="EMPTY"
    )
    model = os.getenv("MODEL_NAME", "finetuned-llama3")

    def predict(inputs):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": inputs["prompt"]}],
            max_tokens=256,
        )
        return {"response": response.choices[0].message.content}

    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=[exact_match_evaluator],
        experiment_prefix="finetuned-eval",
        client=ls_client,
    )
    return results
