from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INVENTORY_SCHEMA_VERSION = "dataset-inventory-1.0"
CONTRACT_SCHEMA_VERSION = "experiment-contract-1.0"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_assets(record: dict[str, Any]) -> list[tuple[str, Path]]:
    assets = []
    for key in ("image_path", "mask_path"):
        value = record.get(key)
        if value:
            assets.append((key.removesuffix("_path"), Path(value)))
    if isinstance(record.get("paths"), dict):
        assets.extend((str(name), Path(value)) for name, value in record["paths"].items())
    return sorted(assets, key=lambda item: item[0])


def _load_records(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("dataset manifest must contain a non-empty records list")
    return payload, records


def build_dataset_inventory(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Hash every manifest asset without embedding machine-specific absolute paths."""
    manifest_path = Path(manifest_path)
    _, records = _load_records(manifest_path)
    seen_sample_ids: set[str] = set()
    entries = []
    for record in sorted(records, key=lambda item: str(item.get("sample_id", ""))):
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in seen_sample_ids:
            raise ValueError("dataset records require unique non-empty sample_id values")
        seen_sample_ids.add(sample_id)
        assets = _record_assets(record)
        if not assets:
            raise ValueError(f"dataset record has no supported assets: {sample_id}")
        inventory_assets = []
        for label, path in assets:
            if not path.is_file():
                raise FileNotFoundError(f"missing dataset asset for {sample_id}/{label}: {path}")
            if path.is_symlink():
                raise ValueError(f"dataset assets cannot be symbolic links: {path}")
            inventory_assets.append(
                {
                    "label": label,
                    "file_name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        entries.append(
            {
                "sample_id": sample_id,
                "split": str(record.get("split", "unspecified")),
                "category": str(record.get("category", "unspecified")),
                "defect_type": str(record.get("defect_type", "unspecified")),
                "anomaly": bool(record.get("anomaly", False)),
                "assets": inventory_assets,
            }
        )

    fingerprint_payload = {"records": entries}
    asset_count = sum(len(entry["assets"]) for entry in entries)
    total_bytes = sum(
        asset["bytes"] for entry in entries for asset in entry["assets"]
    )
    payload = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_fingerprint": canonical_sha256(fingerprint_payload),
        "records": entries,
        "report": {
            "samples": len(entries),
            "assets": asset_count,
            "total_bytes": total_bytes,
            "splits": dict(sorted(Counter(entry["split"] for entry in entries).items())),
            "categories": dict(
                sorted(Counter(entry["category"] for entry in entries).items())
            ),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def verify_dataset_inventory(inventory_path: Path, manifest_path: Path) -> dict[str, Any]:
    expected = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    if expected.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise ValueError("unsupported dataset inventory schema")
    with tempfile.TemporaryDirectory() as directory:
        actual = build_dataset_inventory(
            manifest_path,
            Path(directory) / "verification.json",
        )
    checks = {
        "source_manifest_bytes": actual["manifest_sha256"]
        == expected.get("manifest_sha256"),
        "dataset_fingerprint": actual["dataset_fingerprint"]
        == expected.get("dataset_fingerprint"),
        "records": actual["records"] == expected.get("records"),
    }
    portable_checks = {
        key: checks[key] for key in ("dataset_fingerprint", "records")
    }
    return {
        "pass": all(portable_checks.values()),
        "checks": checks,
        "portable_checks": portable_checks,
        "source_manifest_relocated": not checks["source_manifest_bytes"]
        and all(portable_checks.values()),
        "expected_fingerprint": expected.get("dataset_fingerprint"),
        "actual_fingerprint": actual["dataset_fingerprint"],
        "samples": actual["report"]["samples"],
        "assets": actual["report"]["assets"],
    }


def _git_output(repository_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def capture_code_state(repository_root: Path) -> dict[str, Any]:
    repository_root = Path(repository_root)
    import os

    commit = _git_output(repository_root, "rev-parse", "HEAD") or os.environ.get(
        "SURFACE_SOURCE_COMMIT"
    )
    branch = _git_output(repository_root, "branch", "--show-current") or os.environ.get(
        "SURFACE_SOURCE_BRANCH"
    )
    status = _git_output(repository_root, "status", "--porcelain")
    diff = _git_output(repository_root, "diff", "--binary", "HEAD")
    untracked = sorted(
        line[3:]
        for line in (status or "").splitlines()
        if line.startswith("?? ")
    )
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "tracked_diff_sha256": hashlib.sha256((diff or "").encode("utf-8")).hexdigest(),
        "untracked_files": untracked,
    }


def _dependency_versions(names: list[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_experiment_contract(
    config_path: Path,
    inventory_path: Path,
    output_path: Path,
    *,
    repository_root: Path,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "production-experiment-1.0":
        raise ValueError("unsupported production experiment config schema")
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    if inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise ValueError("unsupported dataset inventory schema")
    code = capture_code_state(repository_root)
    require_clean = bool(config.get("provenance", {}).get("require_clean_git", True))
    require_commit = bool(config.get("provenance", {}).get("require_code_commit", True))
    if require_commit and not code["commit"]:
        raise ValueError(
            "experiment config requires a source commit; run in Git or provide "
            "SURFACE_SOURCE_COMMIT in the container build"
        )
    if require_clean and code["dirty"] and not allow_dirty:
        raise ValueError(
            "experiment config requires a clean Git state; commit/stash changes or allow dirty "
            "development capture explicitly"
        )
    identity = {
        "config_sha256": sha256_file(config_path),
        "dataset_fingerprint": inventory["dataset_fingerprint"],
        "code_commit": code["commit"],
        "tracked_diff_sha256": code["tracked_diff_sha256"],
    }
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "reproducibility_fingerprint": canonical_sha256(identity),
        "identity": identity,
        "configuration": config,
        "dataset": {
            "inventory_schema": inventory["schema_version"],
            "fingerprint": inventory["dataset_fingerprint"],
            "samples": inventory["report"]["samples"],
            "assets": inventory["report"]["assets"],
            "splits": inventory["report"]["splits"],
        },
        "code": code,
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "dependencies": _dependency_versions(
                ["numpy", "Pillow", "torch", "onnx", "onnxruntime", "opencv-python-headless"]
            ),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
