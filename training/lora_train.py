"""LoRA fine-tuning script using PEFT + TRL SFTTrainer."""
import yaml
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import wandb
import os
from dotenv import load_dotenv

load_dotenv()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path: str):
    cfg = load_config(config_path)
    wandb.init(project=os.getenv("WANDB_PROJECT", "llm-finetuning-lab"))

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.eos_token

    qlora_cfg = cfg.get("quantization")
    bnb_config = None
    if qlora_cfg and qlora_cfg.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules"),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    train_cfg = cfg.get("training", {})
    dataset = load_dataset("json", data_files=os.getenv("DATA_PATH", "./data/train.jsonl"), split="train")

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        max_seq_length=train_cfg.get("max_seq_length", 2048),
        report_to=train_cfg.get("report_to", "wandb"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    print(f"Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    args = parser.parse_args()
    train(args.config)
