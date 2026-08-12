from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .reproducibility import sha256_file, verify_dataset_inventory


SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: str) -> str:
    result = SAFE_COMPONENT.sub("-", value).strip("-.")
    if not result or result in {".", ".."}:
        raise ValueError(f"unsafe object-key component: {value!r}")
    return result


def _validate_endpoint_url(endpoint_url: str | None) -> str | None:
    if endpoint_url is None:
        return None
    parsed = urlparse(endpoint_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "S3-compatible endpoint must be an HTTPS URL without credentials, "
            "query parameters, or a fragment"
        )
    return endpoint_url.rstrip("/")


def _record_asset_paths(manifest_path: Path) -> dict[tuple[str, str], Path]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    mapping = {}
    for record in payload.get("records", []):
        sample_id = str(record["sample_id"])
        for key in ("image_path", "mask_path"):
            if record.get(key):
                mapping[(sample_id, key.removesuffix("_path"))] = Path(record[key])
        if isinstance(record.get("paths"), dict):
            for label, path in record["paths"].items():
                mapping[(sample_id, str(label))] = Path(path)
    return mapping


def build_s3_upload_plan(
    manifest_path: Path,
    inventory_path: Path,
    *,
    bucket: str,
    prefix: str = "surface-perception",
    server_side_encryption: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, Any]:
    if not bucket or any(character.isspace() for character in bucket):
        raise ValueError("bucket must be a non-empty name without whitespace")
    if server_side_encryption not in {None, "AES256", "aws:kms"}:
        raise ValueError("server-side encryption must be AES256 or aws:kms")
    endpoint_url = _validate_endpoint_url(endpoint_url)
    verification = verify_dataset_inventory(inventory_path, manifest_path)
    if not verification["pass"]:
        raise ValueError("dataset inventory verification failed; cloud upload refused")
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    local_paths = _record_asset_paths(manifest_path)
    prefix_parts = [_safe_component(part) for part in prefix.split("/") if part]
    if not prefix_parts:
        raise ValueError("prefix must contain at least one safe component")
    clean_prefix = str(PurePosixPath(*prefix_parts))
    fingerprint = inventory["dataset_fingerprint"]
    objects = []
    for record in inventory["records"]:
        sample_id = record["sample_id"]
        for asset in record["assets"]:
            label = asset["label"]
            local_path = local_paths[(sample_id, label)]
            suffix = local_path.suffix.lower()
            key = PurePosixPath(
                clean_prefix,
                "datasets",
                fingerprint,
                _safe_component(record["split"]),
                _safe_component(sample_id),
                f"{_safe_component(label)}{suffix}",
            ).as_posix()
            objects.append(
                {
                    "bucket": bucket,
                    "key": key,
                    "sample_id": sample_id,
                    "label": label,
                    "bytes": asset["bytes"],
                    "sha256": asset["sha256"],
                    "local_path": str(local_path.resolve()),
                }
            )
    return {
        "schema_version": "s3-upload-plan-1.0",
        "mode": "upload-only-no-delete",
        "bucket": bucket,
        "prefix": clean_prefix,
        "dataset_fingerprint": fingerprint,
        "server_side_encryption": server_side_encryption,
        "endpoint_url": endpoint_url,
        "objects": objects,
        "report": {
            "objects": len(objects),
            "bytes": sum(item["bytes"] for item in objects),
            "inventory_verified": True,
        },
    }


def execute_s3_upload_plan(plan: dict[str, Any], *, client=None, dry_run: bool = True) -> dict:
    if plan.get("mode") != "upload-only-no-delete":
        raise ValueError("unsupported or unsafe S3 synchronization mode")
    if client is None and not dry_run:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "Install the cloud extra with 'python -m pip install -e .[cloud]'."
            ) from exc
        client_options = {}
        if plan.get("endpoint_url"):
            client_options["endpoint_url"] = plan["endpoint_url"]
        client = boto3.client("s3", **client_options)

    uploaded = 0
    skipped = 0
    planned = 0
    for item in plan["objects"]:
        if dry_run:
            planned += 1
            continue
        local_path = Path(item["local_path"])
        if local_path.stat().st_size != item["bytes"] or sha256_file(local_path) != item["sha256"]:
            raise ValueError(f"local asset changed after upload planning: {local_path}")
        try:
            response = client.head_object(Bucket=item["bucket"], Key=item["key"])
            remote_hash = response.get("Metadata", {}).get("sha256")
        except Exception as error:  # boto3 exception types are optional at import time
            status = getattr(error, "response", {}).get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            )
            if status != 404:
                raise
            remote_hash = None
        if remote_hash == item["sha256"]:
            skipped += 1
            continue
        extra_args = {
            "Metadata": {
                "sha256": item["sha256"],
                "dataset-fingerprint": plan["dataset_fingerprint"],
            }
        }
        if plan.get("server_side_encryption"):
            extra_args["ServerSideEncryption"] = plan["server_side_encryption"]
        client.upload_file(
            item["local_path"],
            item["bucket"],
            item["key"],
            ExtraArgs=extra_args,
        )
        uploaded += 1
    return {
        "dry_run": dry_run,
        "planned": planned,
        "uploaded": uploaded,
        "skipped_matching": skipped,
        "deleted": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and upload a dataset to S3-compatible storage"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="surface-perception")
    parser.add_argument("--sse", choices=("AES256", "aws:kms"))
    parser.add_argument(
        "--endpoint-url",
        help="HTTPS endpoint for a private S3-compatible service; omit for AWS S3",
    )
    parser.add_argument(
        "--execute", action="store_true", help="upload objects; default is dry-run"
    )
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args()
    plan = build_s3_upload_plan(
        args.manifest,
        args.inventory,
        bucket=args.bucket,
        prefix=args.prefix,
        server_side_encryption=args.sse,
        endpoint_url=args.endpoint_url,
    )
    if args.plan_output:
        safe_plan = {
            **plan,
            "objects": [
                {key: value for key, value in item.items() if key != "local_path"}
                for item in plan["objects"]
            ],
        }
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(
            json.dumps(safe_plan, indent=2, sort_keys=True), encoding="utf-8"
        )
    result = execute_s3_upload_plan(plan, dry_run=not args.execute)
    print(json.dumps({"plan": plan["report"], "result": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
