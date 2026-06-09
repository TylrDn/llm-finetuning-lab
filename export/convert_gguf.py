"""Convert fine-tuned model to GGUF format for llama.cpp deployment.

Two-step process:
  1. HuggingFace → GGUF F32 via llama.cpp/convert_hf_to_gguf.py
  2. Quantize GGUF to the desired quant type via llama-quantize
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Supported quantisation types
SUPPORTED_QUANT_TYPES = ["Q4_K_M", "Q5_K_M", "Q8_0", "F16"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_llama_cpp(llama_cpp_dir: str) -> Path:
    """Verify that the llama.cpp directory exists and contains required tools.

    Parameters
    ----------
    llama_cpp_dir:
        Path to the checked-out llama.cpp repository.

    Returns
    -------
    Path
        Resolved path to the llama.cpp directory.

    Raises
    ------
    FileNotFoundError
        If the directory or required files are missing.
    """
    cpp_path = Path(llama_cpp_dir).resolve()

    if not cpp_path.exists():
        print(
            f"\n[ERROR] llama.cpp directory not found: {cpp_path}\n\n"
            "To fix this, clone and build llama.cpp:\n\n"
            "  git clone https://github.com/ggerganov/llama.cpp.git\n"
            "  cd llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)\n\n"  # noqa: E501
            "Then set LLAMA_CPP_DIR or pass --llama-cpp-dir pointing to the repo root.\n"
        )
        raise FileNotFoundError(f"llama.cpp directory not found: {cpp_path}")

    convert_script = cpp_path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try legacy name
        legacy = cpp_path / "convert.py"
        if legacy.exists():
            logger.warning(
                "convert_hf_to_gguf.py not found; falling back to legacy convert.py. "
                "Consider upgrading llama.cpp."
            )
        else:
            print(
                f"\n[ERROR] Neither convert_hf_to_gguf.py nor convert.py found in {cpp_path}.\n"
                "Ensure you are using a recent version of llama.cpp and that it has been built.\n"
            )
            raise FileNotFoundError(f"Conversion script not found in {cpp_path}")

    quantize_bin = cpp_path / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = cpp_path / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print(
            f"\n[ERROR] llama-quantize binary not found in {cpp_path}.\n"
            "Build llama.cpp first:\n\n"
            "  cd llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)\n"
        )
        raise FileNotFoundError(f"llama-quantize not found in {cpp_path}")

    return cpp_path


def _file_size_mb(path: str | Path) -> float:
    """Return file size in megabytes."""
    size_bytes = Path(path).stat().st_size
    return size_bytes / (1024 ** 2)


def _run(cmd: list[str], description: str) -> None:
    """Run a subprocess command, raising on non-zero exit."""
    logger.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(str(c) for c in cmd)}"
        )


# ---------------------------------------------------------------------------
# Conversion steps
# ---------------------------------------------------------------------------


def convert_to_f32_gguf(
    model_dir: str,
    output_dir: str,
    model_name: str,
    llama_cpp_dir: str,
) -> Path:
    """Convert a HuggingFace model directory to GGUF F32.

    Parameters
    ----------
    model_dir:
        Path to the HuggingFace model directory (safetensors or bin).
    output_dir:
        Directory where the GGUF file will be written.
    model_name:
        Base name for output files (without extension).
    llama_cpp_dir:
        Path to the llama.cpp repository root.

    Returns
    -------
    Path
        Path to the generated F32 GGUF file.
    """
    cpp_path = _check_llama_cpp(llama_cpp_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prefer new conversion script name
    convert_script = cpp_path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = cpp_path / "convert.py"

    f32_gguf = output_path / f"{model_name}.f32.gguf"

    logger.info("Converting HF model to GGUF F32: %s → %s", model_dir, f32_gguf)
    _run(
        [
            sys.executable,
            str(convert_script),
            str(model_dir),
            "--outtype", "f32",
            "--outfile", str(f32_gguf),
        ],
        description="HF → GGUF F32 conversion",
    )

    size_mb = _file_size_mb(f32_gguf)
    logger.info("F32 GGUF created: %s (%.1f MB)", f32_gguf, size_mb)
    print(f"[INFO] F32 GGUF: {f32_gguf} ({size_mb:.1f} MB)")
    return f32_gguf


def quantize_gguf(
    f32_gguf_path: str | Path,
    output_dir: str,
    model_name: str,
    quant_type: str,
    llama_cpp_dir: str,
) -> Path:
    """Quantize a GGUF F32 file to the given quantisation type.

    Parameters
    ----------
    f32_gguf_path:
        Path to the F32 GGUF file.
    output_dir:
        Directory where the quantised GGUF will be written.
    model_name:
        Base name for output files (without extension).
    quant_type:
        Quantisation type, e.g. Q4_K_M. Must be in SUPPORTED_QUANT_TYPES.
    llama_cpp_dir:
        Path to the llama.cpp repository root.

    Returns
    -------
    Path
        Path to the quantised GGUF file.
    """
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"Unsupported quant_type '{quant_type}'. "
            f"Supported: {SUPPORTED_QUANT_TYPES}"
        )

    cpp_path = _check_llama_cpp(llama_cpp_dir)

    quantize_bin = cpp_path / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = cpp_path / "build" / "bin" / "llama-quantize"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    quantized_gguf = output_path / f"{model_name}.{quant_type}.gguf"

    f32_size_mb = _file_size_mb(f32_gguf_path)
    logger.info(
        "Quantising GGUF: %s → %s (type=%s)",
        f32_gguf_path, quantized_gguf, quant_type,
    )
    print(f"[INFO] Input F32 GGUF: {f32_gguf_path} ({f32_size_mb:.1f} MB)")

    _run(
        [
            str(quantize_bin),
            str(f32_gguf_path),
            str(quantized_gguf),
            quant_type,
        ],
        description=f"GGUF quantisation ({quant_type})",
    )

    quantized_size_mb = _file_size_mb(quantized_gguf)
    compression_ratio = f32_size_mb / quantized_size_mb if quantized_size_mb > 0 else float("inf")

    logger.info(
        "Quantised GGUF: %s (%.1f MB, %.1fx compression vs F32)",
        quantized_gguf, quantized_size_mb, compression_ratio,
    )
    print(
        f"[INFO] Quantised GGUF ({quant_type}): {quantized_gguf} "
        f"({quantized_size_mb:.1f} MB, {compression_ratio:.1f}x compression)"
    )
    return quantized_gguf


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------


def convert_model(
    model_dir: str,
    output_dir: str,
    model_name: str,
    quant_type: str = "Q4_K_M",
    llama_cpp_dir: str | None = None,
    keep_f32: bool = False,
) -> dict[str, str]:
    """Full HF → GGUF F32 → quantised GGUF pipeline.

    Parameters
    ----------
    model_dir:
        Path to the HuggingFace model directory.
    output_dir:
        Directory to write output GGUF files.
    model_name:
        Base name for output files.
    quant_type:
        Quantisation type (default Q4_K_M).
    llama_cpp_dir:
        Path to llama.cpp repo. Falls back to LLAMA_CPP_DIR env var, then './llama.cpp'.
    keep_f32:
        If False (default), delete the intermediate F32 GGUF after quantisation.

    Returns
    -------
    dict
        {'f32_gguf': path, 'quantized_gguf': path}
    """
    if llama_cpp_dir is None:
        llama_cpp_dir = os.getenv("LLAMA_CPP_DIR", "./llama.cpp")

    logger.info(
        "Starting GGUF conversion: model=%s, quant=%s, llama_cpp=%s",
        model_name, quant_type, llama_cpp_dir,
    )

    f32_gguf = convert_to_f32_gguf(model_dir, output_dir, model_name, llama_cpp_dir)
    quantized_gguf = quantize_gguf(f32_gguf, output_dir, model_name, quant_type, llama_cpp_dir)

    if not keep_f32 and f32_gguf.exists():
        logger.info("Removing intermediate F32 GGUF: %s", f32_gguf)
        f32_gguf.unlink()

    print("\n[INFO] Conversion complete.")
    print(f"  Output: {quantized_gguf}")
    return {
        "f32_gguf": str(f32_gguf) if keep_f32 else None,
        "quantized_gguf": str(quantized_gguf),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a fine-tuned HuggingFace model to GGUF for llama.cpp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the GGUF output files.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Base name for output files (e.g. 'llama3-8b-dpo').",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="Q4_K_M",
        choices=SUPPORTED_QUANT_TYPES,
        help="GGUF quantisation type.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=str,
        default=os.getenv("LLAMA_CPP_DIR", "./llama.cpp"),
        help="Path to the llama.cpp repository root.",
    )
    parser.add_argument(
        "--keep-f32",
        action="store_true",
        default=False,
        help="Keep the intermediate F32 GGUF file instead of deleting it.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    convert_model(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        quant_type=args.quant_type,
        llama_cpp_dir=args.llama_cpp_dir,
        keep_f32=args.keep_f32,
    )
