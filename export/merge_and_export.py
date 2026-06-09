"""Merge LoRA adapters into base model and export."""
import argparse
import os

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


def merge_and_save(base_model: str, adapter_path: str, output_path: str):
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="cpu")  # noqa: E501

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="./outputs/merged")
    args = parser.parse_args()
    merge_and_save(args.base, args.adapter, args.output)
