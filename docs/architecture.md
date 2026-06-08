# Architecture — llm-finetuning-lab

## Training Pipeline

```mermaid
graph TD
    Raw[Raw Dataset] --> Prep[prepare_dataset.py]
    Prep --> JSONL[train.jsonl / ChatML]
    JSONL --> LoRA[LoRA Train]
    JSONL --> QLoRA[QLoRA Train]
    JSONL --> NeMo[NeMo SFT]
    LoRA --> Adapter[LoRA Adapter]
    QLoRA --> Adapter
    NeMo --> Adapter
    Adapter --> Merge[merge_and_export.py]
    Merge --> Merged[Merged HF Model]
    Merged --> GGUF[GGUF Export]
    Merged --> vLLM[vLLM Serve]
    vLLM --> Eval[LangSmith Eval]
    Eval --> WandB[W&B Dashboard]
```

## Technique Comparison
| Technique | VRAM Required | Speed | Quality |
|---|---|---|---|
| Full FT | 80GB+ | Slow | Best |
| LoRA | 24GB+ | Fast | Near-full |
| QLoRA | 12GB+ | Medium | Good |
| NeMo SFT | 40GB+ | Fast | Best (enterprise) |
