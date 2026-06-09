"""Dataset curation and formatting for SFT."""
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def format_alpaca(example: dict) -> dict:
    """Format to Alpaca instruction-response format."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": text}


def format_chatml(example: dict) -> dict:
    """Format to ChatML format."""
    messages = example.get("messages", [])
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return {"text": text}


def prepare_alpaca_dataset(output_path: str = "./data/train.jsonl"):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    formatted = ds.map(format_alpaca)
    formatted.to_json(output_path)
    print(f"Saved {len(formatted)} examples to {output_path}")
    return formatted


if __name__ == "__main__":
    prepare_alpaca_dataset()
