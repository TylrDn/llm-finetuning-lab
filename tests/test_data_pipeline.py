"""Unit tests for data pipeline."""
from data.prepare_dataset import format_alpaca, format_chatml


def test_format_alpaca_with_input():
    example = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
    result = format_alpaca(example)
    assert "### Instruction:" in result["text"]
    assert "### Input:" in result["text"]
    assert "Hola" in result["text"]


def test_format_alpaca_without_input():
    example = {"instruction": "What is AI?", "input": "", "output": "Artificial Intelligence."}
    result = format_alpaca(example)
    assert "### Input:" not in result["text"]
    assert "Artificial Intelligence" in result["text"]


def test_format_chatml():
    example = {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"}
    ]}
    result = format_chatml(example)
    assert "<|im_start|>user" in result["text"]
    assert "<|im_end|>" in result["text"]
