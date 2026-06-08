# llm-finetuning-lab

End-to-end LLM fine-tuning lab covering full fine-tuning, LoRA, QLoRA, and NVIDIA NeMo-based supervised fine-tuning. Includes dataset curation pipelines, training configs, W&B experiment tracking, and export to NIM/vLLM-compatible formats.

## Techniques Covered
- **LoRA** — parameter-efficient fine-tuning via low-rank adapters
- **QLoRA** — 4-bit quantized LoRA for consumer GPU fine-tuning
- **NeMo SFT** — NVIDIA NeMo supervised fine-tuning pipeline
- **Full Fine-Tuning** — baseline comparison path
- **ORPO / DPO** — preference optimization stubs

## Structure
```
llm-finetuning-lab/
├── data/                # Dataset curation and formatting scripts
├── training/            # Training scripts (LoRA, QLoRA, full)
├── nemo/                # NeMo SFT configs and launcher
├── configs/             # Training hyperparameter configs
├── export/              # Merge adapters + export to GGUF/vLLM
├── evals/               # Post-training eval with LangSmith + lm-eval
├── notebooks/           # Exploratory fine-tuning notebooks
├── deploy/              # Docker + K8s for training jobs
├── tests/               # Unit tests
└── docs/                # Architecture and experiment docs
```

## Quick Start
```bash
cp .env.template .env
pip install -r requirements.txt
python training/lora_train.py --config configs/lora.yaml
```
