from __future__ import annotations

import gc
import importlib
import inspect
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from .modeling import (
    attach_new_lora_adapter,
    bf16_supported,
    load_base_model,
    load_model_with_adapter,
    save_model_metadata,
)
from .utils import ensure_dir, resolve_path, set_global_seed, write_json


def _filtered_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj.__init__)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _strategy_kwargs(args_class, evaluation_strategy: str, save_strategy: str) -> dict[str, Any]:
    parameters = inspect.signature(args_class.__init__).parameters
    kwargs: dict[str, Any] = {}
    if "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = evaluation_strategy
    elif "eval_strategy" in parameters:
        kwargs["eval_strategy"] = evaluation_strategy
    if "save_strategy" in parameters:
        kwargs["save_strategy"] = save_strategy
    if "do_eval" in parameters:
        kwargs["do_eval"] = evaluation_strategy != "no"
    return kwargs


def _trainer_device_flags() -> dict[str, bool]:
    use_bf16 = bf16_supported()
    return {"bf16": use_bf16, "fp16": False}


def _tracking_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("tracking", {}))


def _tracking_enabled(config: dict[str, Any]) -> bool:
    tracking_cfg = _tracking_config(config)
    return bool(tracking_cfg.get("enabled", False) and os.environ.get("COMET_API_KEY"))


def _tracking_project_name(config: dict[str, Any]) -> str:
    tracking_cfg = _tracking_config(config)
    return str(tracking_cfg.get("project_name") or config["project"]["study_name"])


def _tracking_run_name(config: dict[str, Any], run_dir: Path) -> str:
    repo_root = resolve_path(config.get("_repo_root", "."))
    try:
        relative = run_dir.resolve().relative_to(repo_root)
    except ValueError:
        relative = run_dir.resolve()
    return "__".join(relative.parts)


def _trainer_report_to(config: dict[str, Any]) -> list[str]:
    return ["comet_ml"] if _tracking_enabled(config) else []


def _configure_tracking_environment(config: dict[str, Any]) -> None:
    if not _tracking_enabled(config):
        return
    tracking_cfg = _tracking_config(config)
    os.environ.setdefault("COMET_PROJECT_NAME", _tracking_project_name(config))
    if tracking_cfg.get("workspace"):
        os.environ.setdefault("COMET_WORKSPACE", str(tracking_cfg["workspace"]))
    os.environ.setdefault("COMET_MODE", str(tracking_cfg.get("mode", "create")))
    os.environ.setdefault("COMET_START_ONLINE", "1" if bool(tracking_cfg.get("start_online", True)) else "0")
    os.environ.setdefault("COMET_LOG_ASSETS", "TRUE" if bool(tracking_cfg.get("log_assets", True)) else "FALSE")


def _cleanup_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _patch_transformers_cache_symbol() -> None:
    """Restore a legacy cache constant expected by llm_blender on newer transformers."""
    try:
        from transformers.utils import hub as hub_utils
    except Exception:
        return

    if hasattr(hub_utils, "TRANSFORMERS_CACHE"):
        return

    cache_path: str | None = None
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_path = str(HF_HUB_CACHE)
    except Exception:
        pass

    if cache_path is None:
        hf_home = os.environ.get("HF_HOME")
        cache_root = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
        cache_path = str(cache_root / "hub")

    setattr(hub_utils, "TRANSFORMERS_CACHE", cache_path)


def _install_optional_trl_stubs() -> None:
    """Stub TRL optional integrations that this pipeline does not use."""

    try:
        import mergekit.config  # type: ignore  # noqa: F401
    except Exception:
        mergekit_module = sys.modules.get("mergekit")
        if mergekit_module is None:
            mergekit_module = types.ModuleType("mergekit")
            mergekit_module.__path__ = []  # type: ignore[attr-defined]
            sys.modules["mergekit"] = mergekit_module
        elif not hasattr(mergekit_module, "__path__"):
            mergekit_module.__path__ = []  # type: ignore[attr-defined]
        config_module = types.ModuleType("mergekit.config")
        merge_module = types.ModuleType("mergekit.merge")

        class MergeConfiguration:  # noqa: D401
            """Placeholder for TRL optional mergekit integration."""

        class MergeOptions:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        def run_merge(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("mergekit merging is not available in this pipeline.")

        config_module.MergeConfiguration = MergeConfiguration
        merge_module.MergeOptions = MergeOptions
        merge_module.run_merge = run_merge
        mergekit_module.config = config_module
        mergekit_module.merge = merge_module
        sys.modules["mergekit.config"] = config_module
        sys.modules["mergekit.merge"] = merge_module

    try:
        import llm_blender  # type: ignore  # noqa: F401
    except Exception:
        llm_blender_module = types.ModuleType("llm_blender")

        class Blender:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("llm_blender is not available in this pipeline.")

        llm_blender_module.Blender = Blender
        sys.modules["llm_blender"] = llm_blender_module

    try:
        import weave  # type: ignore  # noqa: F401
    except Exception:
        weave_module = types.ModuleType("weave")
        weave_module.__path__ = []  # type: ignore[attr-defined]
        trace_module = types.ModuleType("weave.trace")
        trace_module.__path__ = []  # type: ignore[attr-defined]
        context_module = types.ModuleType("weave.trace.context")
        context_module.__path__ = []  # type: ignore[attr-defined]
        weave_client_context_module = types.ModuleType("weave.trace.context.weave_client_context")

        class EvaluationLogger:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

            def log(self, *args: Any, **kwargs: Any) -> None:
                return None

        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        def _missing_weave(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("weave callbacks are not enabled in this pipeline.")

        def _missing_weave_attr(name: str) -> Any:
            if name.startswith("__"):
                raise AttributeError(name)
            raise RuntimeError(f"weave attribute `{name}` is not available in this pipeline.")

        weave_client_context_module.get_weave_client = _noop
        weave_client_context_module.require_weave_client = _noop
        weave_client_context_module.set_weave_client_global = _noop
        weave_client_context_module.reset_weave_client_global = _noop
        context_module.weave_client_context = weave_client_context_module
        trace_module.context = context_module
        weave_module.EvaluationLogger = EvaluationLogger
        weave_module.init = _missing_weave
        weave_module.publish = _missing_weave
        weave_module.trace = trace_module
        weave_module.__getattr__ = _missing_weave_attr  # type: ignore[attr-defined]
        sys.modules["weave"] = weave_module
        sys.modules["weave.trace"] = trace_module
        sys.modules["weave.trace.context"] = context_module
        sys.modules["weave.trace.context.weave_client_context"] = weave_client_context_module


def _ensure_model_warning_registry(model: Any) -> None:
    """Add `warnings_issued` to wrapped models expected by recent TRL releases."""
    visited: set[int] = set()
    stack = [model]

    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        try:
            existing = getattr(current, "warnings_issued")
        except Exception:
            existing = None

        if not isinstance(existing, dict):
            try:
                setattr(current, "warnings_issued", {})
            except Exception:
                pass

        for attr_name in ("base_model", "model"):
            try:
                nested = getattr(current, attr_name)
            except Exception:
                nested = None
            if nested is not None and nested is not current:
                stack.append(nested)


def _import_kto_trainer_classes():
    errors: list[Exception] = []
    for module_name in ["trl", "trl.experimental.kto"]:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "KTOTrainer"), getattr(module, "KTOConfig")
        except Exception as exc:
            errors.append(exc)
    raise RuntimeError(
        "Could not import KTOTrainer/KTOConfig from TRL. "
        "The installed TRL version may not support KTO."
    ) from errors[-1]


def _disable_multimodal_processing(config: dict[str, Any]) -> bool:
    return bool(config.get("model", {}).get("disable_multimodal_processing", False))


def _iter_model_configs(model: Any):
    visited_models: set[int] = set()
    visited_configs: set[int] = set()
    stack = [model]

    while stack:
        current = stack.pop()
        if current is None or id(current) in visited_models:
            continue
        visited_models.add(id(current))

        try:
            model_config = getattr(current, "config")
        except Exception:
            model_config = None
        if model_config is not None and id(model_config) not in visited_configs:
            visited_configs.add(id(model_config))
            yield model_config

        for attr_name in ("base_model", "model"):
            try:
                nested = getattr(current, attr_name)
            except Exception:
                nested = None
            if nested is not None and nested is not current:
                stack.append(nested)


@contextmanager
def _force_text_only_trainer_path(config: dict[str, Any], *models: Any):
    if not _disable_multimodal_processing(config):
        yield
        return

    original_model_types: list[tuple[Any, Any]] = []
    text_only_suffix = "_text_only"
    for model in models:
        for model_config in _iter_model_configs(model):
            if not hasattr(model_config, "model_type"):
                continue
            try:
                original_model_type = model_config.model_type
                overridden_model_type = f"{original_model_type}{text_only_suffix}"
                model_config.model_type = overridden_model_type
                original_model_types.append((model_config, original_model_type))
            except Exception:
                continue

    try:
        yield
    finally:
        for model_config, original_model_type in reversed(original_model_types):
            try:
                model_config.model_type = original_model_type
            except Exception:
                continue


def _requires_token_type_ids(config: dict[str, Any], model: Any) -> bool:
    if not _disable_multimodal_processing(config):
        return False

    for model_config in _iter_model_configs(model):
        model_type = str(getattr(model_config, "model_type", "")).lower()
        if "gemma3" in model_type:
            return True
    return False


def _patch_missing_token_type_ids(config: dict[str, Any], model: Any) -> Any:
    if not _requires_token_type_ids(config, model):
        return model
    if getattr(model, "_codex_token_type_ids_patched", False):
        return model

    try:
        import torch
    except Exception:
        return model

    original_forward = model.forward

    def patched_forward(self, *args: Any, **kwargs: Any):
        if kwargs.get("token_type_ids") is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]

            if input_ids is not None:
                labels = kwargs.get("labels")
                attention_mask = kwargs.get("attention_mask")

                if (
                    labels is not None
                    and hasattr(labels, "shape")
                    and hasattr(input_ids, "shape")
                    and tuple(labels.shape) == tuple(input_ids.shape)
                ):
                    token_type_ids = labels.ne(-100).to(dtype=torch.long)
                else:
                    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

                if attention_mask is not None and hasattr(attention_mask, "to"):
                    token_type_ids = token_type_ids * attention_mask.to(dtype=torch.long)

                kwargs["token_type_ids"] = token_type_ids

        return original_forward(*args, **kwargs)

    model.forward = types.MethodType(patched_forward, model)
    model._codex_token_type_ids_patched = True  # type: ignore[attr-defined]
    model._codex_original_forward = original_forward  # type: ignore[attr-defined]
    return model


def _save_training_summary(run_dir: Path, result_metrics: dict[str, Any], trainer) -> None:
    history = getattr(getattr(trainer, "state", None), "log_history", [])
    write_json(run_dir / "training_metrics.json", result_metrics)
    write_json(run_dir / "trainer_log_history.json", history)


def _sft_formatting_func(example: dict[str, Any]) -> list[str]:
    text = example["sft_text"]
    if isinstance(text, list):
        return [str(item) for item in text]
    return [str(text)]


def train_sft(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: Path,
    seed: int,
) -> Path:
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    set_global_seed(seed)
    _configure_tracking_environment(config)
    model, tokenizer = load_base_model(config)
    model = attach_new_lora_adapter(model, config, seed=seed)

    train_dataset = Dataset.from_pandas(train_df[["sft_text"]], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[["sft_text"]], preserve_index=False)

    sft_cfg = config["sft"]
    training_args = TrainingArguments(
        output_dir=str(ensure_dir(run_dir / "trainer")),
        num_train_epochs=float(sft_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(sft_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(sft_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(sft_cfg["gradient_accumulation_steps"]),
        learning_rate=float(sft_cfg["learning_rate"]),
        warmup_ratio=float(sft_cfg["warmup_ratio"]),
        weight_decay=float(sft_cfg["weight_decay"]),
        logging_steps=int(sft_cfg["logging_steps"]),
        save_total_limit=int(sft_cfg["save_total_limit"]),
        optim=str(sft_cfg["optim"]),
        seed=seed,
        report_to=_trainer_report_to(config),
        run_name=_tracking_run_name(config, run_dir),
        remove_unused_columns=False,
        **_strategy_kwargs(
            TrainingArguments,
            evaluation_strategy=str(sft_cfg["evaluation_strategy"]),
            save_strategy=str(sft_cfg["save_strategy"]),
        ),
        **_trainer_device_flags(),
    )
    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": test_dataset,
        "dataset_text_field": "sft_text",
        "formatting_func": _sft_formatting_func,
        "max_seq_length": int(config["model"]["max_seq_length"]),
        "packing": bool(sft_cfg["packing"]),
        "dataset_num_proc": 1,
    }
    trainer = SFTTrainer(**_filtered_kwargs(SFTTrainer, trainer_kwargs))
    train_result = trainer.train()

    adapter_dir = ensure_dir(run_dir / "adapter")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    save_model_metadata(config, run_dir, adapter_dir)
    _save_training_summary(run_dir, dict(train_result.metrics), trainer)
    del trainer
    _cleanup_cuda()
    return adapter_dir


def train_dpo(
    config: dict[str, Any],
    preference_df: pd.DataFrame,
    run_dir: Path,
    seed: int,
    start_from: str,
    sft_adapter_dir: str | Path | None = None,
) -> Path:
    from datasets import Dataset
    from transformers import TrainingArguments

    _patch_transformers_cache_symbol()
    _install_optional_trl_stubs()

    try:
        from trl import DPOTrainer
    except Exception as exc:
        error_text = str(exc)
        if "mergekit" in error_text or "llm_blender" in error_text or "llm-blender" in error_text or "weave" in error_text:
            raise RuntimeError(
                "TRL's DPOTrainer is still trying to import optional integrations before the local "
                "compatibility stubs are applied. Make sure Colab is running the latest "
                "`src/topic_alignment/training.py`, then rerun only the DPO commands."
            ) from exc
        if "TRANSFORMERS_CACHE" in error_text:
            raise RuntimeError(
                "TRL's DPOTrainer is importing llm_blender, which expects the legacy "
                "`transformers.utils.hub.TRANSFORMERS_CACHE` symbol. This repo now patches that "
                "compatibility path automatically, so make sure Colab is running the latest copy of "
                "`src/topic_alignment/training.py`, then rerun only the DPO commands."
            ) from exc
        raise

    try:
        from trl import DPOConfig  # type: ignore
    except ImportError:
        DPOConfig = None

    try:
        from unsloth import PatchDPOTrainer

        PatchDPOTrainer()
    except Exception:
        pass

    set_global_seed(seed)
    dpo_cfg = config["dpo"]
    _configure_tracking_environment(config)

    if start_from == "base":
        policy_model, tokenizer = load_base_model(config)
        policy_model = attach_new_lora_adapter(policy_model, config, seed=seed)
        reference_model, _ = load_base_model(config)
    elif start_from == "sft":
        if sft_adapter_dir is None:
            raise ValueError("SFT adapter directory is required for SFT->DPO training.")
        policy_model, tokenizer = load_model_with_adapter(config, sft_adapter_dir, trainable=True)
        reference_model, _ = load_model_with_adapter(config, sft_adapter_dir, trainable=False)
    else:
        raise ValueError(f"Unsupported DPO start point: {start_from}")

    _ensure_model_warning_registry(policy_model)
    _ensure_model_warning_registry(reference_model)
    policy_model = _patch_missing_token_type_ids(config, policy_model)
    reference_model = _patch_missing_token_type_ids(config, reference_model)

    dataset = Dataset.from_pandas(preference_df[["prompt", "chosen", "rejected"]], preserve_index=False)

    common_args = {
        "output_dir": str(ensure_dir(run_dir / "trainer")),
        "num_train_epochs": float(dpo_cfg["num_train_epochs"]),
        "per_device_train_batch_size": int(dpo_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(dpo_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(dpo_cfg["learning_rate"]),
        "warmup_ratio": float(dpo_cfg["warmup_ratio"]),
        "weight_decay": float(dpo_cfg["weight_decay"]),
        "optim": str(dpo_cfg.get("optim", "adamw_torch")),
        "logging_steps": int(dpo_cfg["logging_steps"]),
        "save_total_limit": int(dpo_cfg["save_total_limit"]),
        "seed": seed,
        "report_to": _trainer_report_to(config),
        "run_name": _tracking_run_name(config, run_dir),
        "remove_unused_columns": False,
        **_trainer_device_flags(),
    }

    trainer_specific_kwargs = {
        "model": policy_model,
        "ref_model": reference_model,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "train_dataset": dataset,
        "eval_dataset": None,
    }

    if DPOConfig is not None:
        training_args = DPOConfig(
            beta=float(dpo_cfg["beta"]),
            loss_type=str(dpo_cfg["loss_type"]),
            max_prompt_length=int(dpo_cfg["max_prompt_length"]),
            max_length=int(dpo_cfg["max_length"]),
            **_strategy_kwargs(
                DPOConfig,
                evaluation_strategy=str(dpo_cfg["evaluation_strategy"]),
                save_strategy=str(dpo_cfg["save_strategy"]),
            ),
            **common_args,
        )
        trainer_kwargs = trainer_specific_kwargs | {"args": training_args}
    else:
        training_args = TrainingArguments(
            **_strategy_kwargs(
                TrainingArguments,
                evaluation_strategy=str(dpo_cfg["evaluation_strategy"]),
                save_strategy=str(dpo_cfg["save_strategy"]),
            ),
            **common_args,
        )
        trainer_kwargs = trainer_specific_kwargs | {
            "args": training_args,
            "beta": float(dpo_cfg["beta"]),
            "loss_type": str(dpo_cfg["loss_type"]),
            "max_prompt_length": int(dpo_cfg["max_prompt_length"]),
            "max_length": int(dpo_cfg["max_length"]),
        }

    with _force_text_only_trainer_path(config, policy_model, reference_model):
        trainer = DPOTrainer(**_filtered_kwargs(DPOTrainer, trainer_kwargs))
        train_result = trainer.train()

        adapter_dir = ensure_dir(run_dir / "adapter")
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        save_model_metadata(config, run_dir, adapter_dir)
        _save_training_summary(run_dir, dict(train_result.metrics), trainer)
    del trainer
    _cleanup_cuda()
    return adapter_dir


def train_kto(
    config: dict[str, Any],
    kto_df: pd.DataFrame,
    run_dir: Path,
    seed: int,
    start_from: str,
    sft_adapter_dir: str | Path | None = None,
) -> Path:
    from datasets import Dataset

    _patch_transformers_cache_symbol()
    _install_optional_trl_stubs()

    try:
        KTOTrainer, KTOConfig = _import_kto_trainer_classes()
    except Exception as exc:
        error_text = str(exc)
        if "mergekit" in error_text or "llm_blender" in error_text or "llm-blender" in error_text or "weave" in error_text:
            raise RuntimeError(
                "TRL's KTOTrainer is still trying to import optional integrations before the local "
                "compatibility stubs are applied. Make sure Colab is running the latest "
                "`src/topic_alignment/training.py`, then rerun only the KTO commands."
            ) from exc
        raise

    set_global_seed(seed)
    _configure_tracking_environment(config)
    kto_cfg = config["kto"]

    if start_from == "base":
        policy_model, tokenizer = load_base_model(config)
        policy_model = attach_new_lora_adapter(policy_model, config, seed=seed)
        reference_model, _ = load_base_model(config)
    elif start_from == "sft":
        if sft_adapter_dir is None:
            raise ValueError("SFT adapter directory is required for SFT->KTO training.")
        policy_model, tokenizer = load_model_with_adapter(config, sft_adapter_dir, trainable=True)
        reference_model, _ = load_model_with_adapter(config, sft_adapter_dir, trainable=False)
    else:
        raise ValueError(f"Unsupported KTO start point: {start_from}")

    _ensure_model_warning_registry(policy_model)
    _ensure_model_warning_registry(reference_model)
    policy_model = _patch_missing_token_type_ids(config, policy_model)
    reference_model = _patch_missing_token_type_ids(config, reference_model)

    dataset = Dataset.from_pandas(kto_df[["prompt", "completion", "label"]], preserve_index=False)

    training_args = KTOConfig(
        output_dir=str(ensure_dir(run_dir / "trainer")),
        num_train_epochs=float(kto_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(kto_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(kto_cfg["gradient_accumulation_steps"]),
        learning_rate=float(kto_cfg["learning_rate"]),
        warmup_ratio=float(kto_cfg["warmup_ratio"]),
        weight_decay=float(kto_cfg["weight_decay"]),
        optim=str(kto_cfg["optim"]),
        beta=float(kto_cfg["beta"]),
        loss_type=str(kto_cfg["loss_type"]),
        desirable_weight=float(kto_cfg["desirable_weight"]),
        undesirable_weight=float(kto_cfg["undesirable_weight"]),
        logging_steps=int(kto_cfg["logging_steps"]),
        save_total_limit=int(kto_cfg["save_total_limit"]),
        max_prompt_length=int(kto_cfg["max_prompt_length"]),
        max_length=int(kto_cfg["max_length"]),
        disable_dropout=bool(kto_cfg.get("disable_dropout", True)),
        seed=seed,
        report_to=_trainer_report_to(config),
        run_name=_tracking_run_name(config, run_dir),
        remove_unused_columns=False,
        dataset_num_proc=1,
        **_strategy_kwargs(
            KTOConfig,
            evaluation_strategy=str(kto_cfg["evaluation_strategy"]),
            save_strategy=str(kto_cfg["save_strategy"]),
        ),
        **_trainer_device_flags(),
    )
    trainer_kwargs = {
        "model": policy_model,
        "ref_model": reference_model,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "args": training_args,
        "train_dataset": dataset,
        "eval_dataset": None,
    }
    with _force_text_only_trainer_path(config, policy_model, reference_model):
        trainer = KTOTrainer(**_filtered_kwargs(KTOTrainer, trainer_kwargs))
        train_result = trainer.train()

        adapter_dir = ensure_dir(run_dir / "adapter")
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        save_model_metadata(config, run_dir, adapter_dir)
        _save_training_summary(run_dir, dict(train_result.metrics), trainer)
    del trainer
    _cleanup_cuda()
    return adapter_dir
