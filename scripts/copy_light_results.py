from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE = Path("/content/drive/MyDrive/beyond_sft_alignment")

SKIP_DIR_NAMES = {
    ".cache",
    ".ipynb_checkpoints",
    "__pycache__",
    "adapter",
    "trainer",
    "wandb",
}
SKIP_DIR_PREFIXES = ("checkpoint-",)
SKIP_FILE_NAMES = {
    "adapter_model.bin",
    "adapter_model.safetensors",
    "model.safetensors",
    "optimizer.pt",
    "pytorch_model.bin",
    "rng_state.pth",
    "scaler.pt",
    "scheduler.pt",
    "training_args.bin",
}
SKIP_FILE_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
)
EVIDENCE_FILE_NAMES = {
    "logprob_predictions_test.csv",
    "predictions_test.csv",
    "run_metadata.json",
    "training_metrics.json",
    "trainer_log_history.json",
}
EVIDENCE_FILE_SUFFIXES = (".json",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a Drive results folder while excluding LoRA adapters, trainer "
            "checkpoints, and other large model artifacts."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Folder to copy. Defaults to {DEFAULT_SOURCE}.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination folder. Defaults to SOURCE with '_light' appended.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be copied or skipped without writing files.",
    )
    parser.add_argument(
        "--strict-evidence",
        action="store_true",
        help="Exit with an error if source prediction/JSON evidence files are missing from the copy.",
    )
    return parser.parse_args()


def default_dest(source: Path) -> Path:
    return source.with_name(f"{source.name}_light_result")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = Path(root) / filename
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def should_skip_dir(path: Path) -> bool:
    name = path.name
    return name in SKIP_DIR_NAMES or any(name.startswith(prefix) for prefix in SKIP_DIR_PREFIXES)


def should_skip_file(path: Path) -> bool:
    name = path.name
    return name in SKIP_FILE_NAMES or path.suffix in SKIP_FILE_SUFFIXES


def should_skip_path(path: Path, source: Path) -> bool:
    rel_path = path.relative_to(source)
    for part in rel_path.parts[:-1]:
        if part in SKIP_DIR_NAMES or any(part.startswith(prefix) for prefix in SKIP_DIR_PREFIXES):
            return True
    return should_skip_file(path)


def is_evidence_file(path: Path) -> bool:
    return path.name in EVIDENCE_FILE_NAMES or path.suffix in EVIDENCE_FILE_SUFFIXES


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def record_skipped(records: list[dict[str, object]], kind: str, rel_path: Path, size: int) -> None:
    records.append(
        {
            "kind": kind,
            "path": rel_path.as_posix(),
            "bytes": size,
        }
    )


def write_manifest(
    dest: Path,
    source: Path,
    copied_files: int,
    copied_bytes: int,
    skipped_records: Iterable[dict[str, object]],
    skipped_bytes: int,
    evidence_records: Iterable[dict[str, object]],
    dry_run: bool,
) -> None:
    records = list(skipped_records)
    evidence = list(evidence_records)
    if dry_run:
        return

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(dest),
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "copied_size": format_bytes(copied_bytes),
        "skipped_items": len(records),
        "skipped_bytes": skipped_bytes,
        "skipped_size": format_bytes(skipped_bytes),
        "skip_dir_names": sorted(SKIP_DIR_NAMES),
        "skip_dir_prefixes": list(SKIP_DIR_PREFIXES),
        "skip_file_names": sorted(SKIP_FILE_NAMES),
        "skip_file_suffixes": list(SKIP_FILE_SUFFIXES),
        "evidence_files_found": len(evidence),
        "evidence_files_copied": sum(1 for record in evidence if record["status"] == "copied"),
        "evidence_files_skipped_by_rule": sum(1 for record in evidence if record["status"] == "skipped_by_rule"),
        "evidence_files_missing_unexpected": sum(1 for record in evidence if record["status"] == "missing_unexpected"),
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "light_copy_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    skipped_text = "\n".join(
        f"{record['kind']}\t{format_bytes(int(record['bytes']))}\t{record['path']}"
        for record in records
    )
    (dest / "light_copy_skipped_paths.txt").write_text(skipped_text + ("\n" if skipped_text else ""), encoding="utf-8")
    (dest / "README_LIGHT_COPY.txt").write_text(
        "This is a lightweight copy of the experiment folder.\n"
        "LoRA adapter directories, trainer directories, checkpoints, and model weight files were excluded.\n"
        "Use the original full folder if you need to reload adapters or resume training.\n",
        encoding="utf-8",
    )
    evidence_text = "\n".join(
        f"{record['status']}\t{record['path']}"
        for record in evidence
    )
    (dest / "light_copy_evidence_report.txt").write_text(
        evidence_text + ("\n" if evidence_text else ""),
        encoding="utf-8",
    )


def build_evidence_report(source: Path, dest: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for root, _, files in os.walk(source):
        root_path = Path(root)
        for filename in files:
            source_file = root_path / filename
            if not is_evidence_file(source_file):
                continue
            rel_path = source_file.relative_to(source)
            dest_file = dest / rel_path
            if dest_file.exists():
                status = "copied"
            elif should_skip_path(source_file, source):
                status = "skipped_by_rule"
            else:
                status = "missing_unexpected"
            records.append({"path": rel_path.as_posix(), "status": status})
    return sorted(records, key=lambda record: (str(record["status"]), str(record["path"])))


def copy_light(source: Path, dest: Path, dry_run: bool) -> dict[str, int]:
    copied_files = 0
    copied_bytes = 0
    skipped_bytes = 0
    skipped_records: list[dict[str, object]] = []

    for root, dirs, files in os.walk(source):
        root_path = Path(root)
        rel_root = root_path.relative_to(source)

        for dirname in list(dirs):
            dir_path = root_path / dirname
            rel_path = dir_path.relative_to(source)
            if should_skip_dir(dir_path):
                size = tree_size(dir_path)
                skipped_bytes += size
                record_skipped(skipped_records, "dir", rel_path, size)
                dirs.remove(dirname)

        target_root = dest / rel_root
        if not dry_run:
            target_root.mkdir(parents=True, exist_ok=True)

        for filename in files:
            source_file = root_path / filename
            rel_path = source_file.relative_to(source)
            size = source_file.stat().st_size
            if should_skip_file(source_file):
                skipped_bytes += size
                record_skipped(skipped_records, "file", rel_path, size)
                continue

            copied_files += 1
            copied_bytes += size
            if not dry_run:
                shutil.copy2(source_file, target_root / filename)

    evidence_records = build_evidence_report(source, dest)
    write_manifest(
        dest=dest,
        source=source,
        copied_files=copied_files,
        copied_bytes=copied_bytes,
        skipped_records=skipped_records,
        skipped_bytes=skipped_bytes,
        evidence_records=evidence_records,
        dry_run=dry_run,
    )
    return {
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "skipped_items": len(skipped_records),
        "skipped_bytes": skipped_bytes,
        "evidence_files_found": len(evidence_records),
        "evidence_files_copied": sum(1 for record in evidence_records if record["status"] == "copied"),
        "evidence_files_skipped_by_rule": sum(1 for record in evidence_records if record["status"] == "skipped_by_rule"),
        "evidence_files_missing_unexpected": sum(1 for record in evidence_records if record["status"] == "missing_unexpected"),
    }


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    dest = (args.dest.expanduser() if args.dest else default_dest(source)).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source is not a folder: {source}")
    if source == dest:
        raise ValueError("Destination must be different from source.")
    if is_relative_to(dest, source):
        raise ValueError("Destination cannot be inside source, otherwise the copy can recurse into itself.")

    summary = copy_light(source, dest, dry_run=args.dry_run)
    action = "dry run complete" if args.dry_run else "copy complete"
    print(action)
    print(f"source: {source}")
    print(f"destination: {dest}")
    print(f"copied files: {summary['copied_files']} ({format_bytes(summary['copied_bytes'])})")
    print(f"skipped items: {summary['skipped_items']} ({format_bytes(summary['skipped_bytes'])})")
    print(
        "evidence files: "
        f"{summary['evidence_files_copied']} copied, "
        f"{summary['evidence_files_skipped_by_rule']} skipped by rule, "
        f"{summary['evidence_files_missing_unexpected']} missing unexpectedly "
        f"out of {summary['evidence_files_found']} found"
    )
    if not args.dry_run:
        print(f"evidence report: {dest / 'light_copy_evidence_report.txt'}")
    if args.strict_evidence and summary["evidence_files_missing_unexpected"]:
        raise SystemExit("Some evidence files were found in the source but are missing from the light copy.")


if __name__ == "__main__":
    main()
