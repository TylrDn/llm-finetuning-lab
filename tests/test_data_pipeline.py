"""Unit tests for data pipeline."""

from unittest.mock import MagicMock, patch

from data.prepare_dataset import format_alpaca, format_chatml, prepare_alpaca_dataset


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


def test_prepare_alpaca_dataset_writes_json(tmp_path):
    mock_ds = MagicMock()
    mock_formatted = MagicMock()
    mock_formatted.__len__.return_value = 2
    mock_ds.map.return_value = mock_formatted
    with patch("data.prepare_dataset.load_dataset", return_value=mock_ds):
        output = tmp_path / "train.jsonl"
        result = prepare_alpaca_dataset(str(output))
    mock_formatted.to_json.assert_called_once_with(str(output))
    assert result is mock_formatted
