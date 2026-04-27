from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_STABLE_DOWNLOADS", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

try:
    import unsloth  # noqa: F401
except Exception:
    unsloth = None

from .config import config_hash
from .utils import ensure_dir, resolve_path, write_json


def _model_backend(config: dict[str, Any]) -> str:
    return str(config.get("model", {}).get("model_backend", "language")).lower()


def _uses_fast_vision_model(config: dict[str, Any]) -> bool:
    return _model_backend(config) == "vision"


def _disable_multimodal_processing(config: dict[str, Any]) -> bool:
    return bool(config.get("model", {}).get("disable_multimodal_processing", False))


def _configured_model_source(config: dict[str, Any]) -> dict[str, str | None]:
    model_cfg = config.get("model", {})
    return {
        "role": "base",
        "model_id": str(model_cfg["base_model_id"]),
    }


def _processing_class_for_text_only(config: dict[str, Any], processing_class: Any) -> Any:
    if _disable_multimodal_processing(config) and hasattr(processing_class, "tokenizer"):
        return processing_class.tokenizer
    return processing_class


def bf16_supported() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def load_base_model(config: dict[str, Any]):
    model_source = _configured_model_source(config)
    if _uses_fast_vision_model(config):
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=str(model_source["model_id"]),
            max_seq_length=int(config["model"]["max_seq_length"]),
            load_in_4bit=bool(config["model"]["load_in_4bit"]),
            dtype=None,
        )
    else:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_source["model_id"]),
            max_seq_length=int(config["model"]["max_seq_length"]),
            load_in_4bit=bool(config["model"]["load_in_4bit"]),
            dtype=None,
        )
    tokenizer = _processing_class_for_text_only(config, tokenizer)
    return model, tokenizer


def attach_new_lora_adapter(model, config: dict[str, Any], seed: int):
    peft_config = config["peft"]
    if _uses_fast_vision_model(config):
        from unsloth import FastVisionModel

        return FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=int(peft_config["r"]),
            lora_alpha=int(peft_config["lora_alpha"]),
            lora_dropout=float(peft_config["lora_dropout"]),
            bias=str(peft_config["bias"]),
            use_gradient_checkpointing=peft_config["gradient_checkpointing"],
            random_state=seed,
        )

    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=int(peft_config["r"]),
        target_modules=list(peft_config["target_modules"]),
        lora_alpha=int(peft_config["lora_alpha"]),
        lora_dropout=float(peft_config["lora_dropout"]),
        bias=str(peft_config["bias"]),
        use_gradient_checkpointing=peft_config["gradient_checkpointing"],
        random_state=seed,
    )


def load_model_with_adapter(config: dict[str, Any], adapter_path: str | Path, trainable: bool):
    from peft import PeftModel

    model, tokenizer = load_base_model(config)
    model = PeftModel.from_pretrained(
        model,
        str(resolve_path(adapter_path)),
        is_trainable=trainable,
    )
    return model, tokenizer


def prepare_for_inference(model) -> Any:
    try:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    except Exception:
        try:
            from unsloth import FastVisionModel

            FastVisionModel.for_inference(model)
        except Exception:
            pass
    model.eval()
    return model


def model_device(model):
    return next(model.parameters()).device


def create_run_dir(config: dict[str, Any], *parts: str) -> Path:
    root = ensure_dir(resolve_path(config["project"]["outputs_root"]) / "runs")
    run_dir = root
    for part in parts:
        run_dir = run_dir / part
    return ensure_dir(run_dir)


def create_stage_run_dir(
    config: dict[str, Any],
    seed: int,
    train_size: int | str,
    data_allocation: str,
    stage_name: str,
) -> Path:
    return create_run_dir(
        config,
        f"seed_{seed}",
        f"train_{str(train_size).lower()}",
        f"allocation_{str(data_allocation).lower()}",
        stage_name,
    )


def save_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    write_json(run_dir / "metadata.json", metadata)


def _repo_relative_path(config: dict[str, Any], path: Path) -> str:
    repo_root = resolve_path(config.get("_repo_root", "."))
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def save_model_metadata(config: dict[str, Any], run_dir: Path, adapter_dir: Path) -> None:
    run_metadata_path = run_dir / "metadata.json"
    run_metadata: dict[str, Any] = {}
    if run_metadata_path.exists():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))

    tracking_cfg = dict(config.get("tracking", {}))
    base_source = _configured_model_source(config)
    payload = {
        "base_model": {
            "family": str(config["model"]["base_family"]),
            "base_model_id": str(base_source["model_id"]),
        },
        "adapter": {
            "format": "peft_lora_adapter",
            "path": _repo_relative_path(config, adapter_dir),
            "run": {
                "path": _repo_relative_path(config, run_dir),
                **run_metadata,
            },
        },
        "quantization": {
            "load_in_4bit": bool(config["model"]["load_in_4bit"]),
            "scheme": "bitsandbytes_4bit" if bool(config["model"]["load_in_4bit"]) else "none",
        },
        "prompt_format": {
            "prompt_version": str(config["format"]["prompt_version"]),
            "response_prefix": str(config["format"]["response_prefix"]),
        },
        "tracking": {
            "enabled": bool(tracking_cfg.get("enabled", False)),
            "provider": str(tracking_cfg.get("provider", "none")),
            "project_name": str(tracking_cfg.get("project_name") or config["project"]["study_name"]),
            "workspace": tracking_cfg.get("workspace"),
            "log_assets": bool(tracking_cfg.get("log_assets", True)),
        },
        "config_hash": str(config.get("_config_hash") or config_hash(config)),
    }
    write_json(adapter_dir / "model_metadata.json", payload)
