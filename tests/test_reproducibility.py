import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from surface_perception.cloud_sync import build_s3_upload_plan, execute_s3_upload_plan
from surface_perception.reproducibility import (
    build_dataset_inventory,
    build_experiment_contract,
    verify_dataset_inventory,
)


def _write_dataset(root: Path) -> Path:
    records = []
    for split, count in (("train", 2), ("validation", 1), ("test", 1)):
        for index in range(count):
            sample_id = f"{split}_{index}"
            image_path = root / "images" / f"{sample_id}.png"
            mask_path = root / "masks" / f"{sample_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((16, 16, 3), 60 + index, dtype=np.uint8)).save(
                image_path
            )
            Image.fromarray(np.eye(16, dtype=np.uint8) * 255).save(mask_path)
            records.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "category": "panel",
                    "defect_type": "scratch",
                    "anomaly": True,
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                }
            )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "1.0", "records": records}),
        encoding="utf-8",
    )
    return manifest_path


class _MissingObjectError(Exception):
    response = {"ResponseMetadata": {"HTTPStatusCode": 404}}


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects = {}
        self.uploads = []

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        value = self.objects.get((Bucket, Key))
        if value is None:
            raise _MissingObjectError()
        return value

    def upload_file(self, path: str, bucket: str, key: str, ExtraArgs: dict) -> None:
        self.uploads.append((path, bucket, key, ExtraArgs))
        self.objects[(bucket, key)] = {"Metadata": ExtraArgs["Metadata"]}


class ProductionReproducibilityTest(unittest.TestCase):
    def test_inventory_is_deterministic_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_dataset(root)
            first_path = root / "first.json"
            second_path = root / "second.json"
            first = build_dataset_inventory(manifest, first_path)
            second = build_dataset_inventory(manifest, second_path)

            self.assertEqual(first, second)
            self.assertEqual(first["report"]["samples"], 4)
            self.assertEqual(first["report"]["assets"], 8)
            self.assertTrue(verify_dataset_inventory(first_path, manifest)["pass"])

            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
                root / "images" / "test_0.png"
            )
            self.assertFalse(verify_dataset_inventory(first_path, manifest)["pass"])

    def test_inventory_accepts_path_relocation_when_bytes_and_records_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            relocated = root / "relocated"
            manifest = _write_dataset(original)
            inventory_path = root / "inventory.json"
            build_dataset_inventory(manifest, inventory_path)
            relocated_manifest = _write_dataset(relocated)

            report = verify_dataset_inventory(inventory_path, relocated_manifest)
            self.assertTrue(report["pass"])
            self.assertTrue(report["source_manifest_relocated"])
            self.assertFalse(report["checks"]["source_manifest_bytes"])

    def test_s3_plan_is_content_addressed_upload_only_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_dataset(root)
            inventory_path = root / "inventory.json"
            inventory = build_dataset_inventory(manifest, inventory_path)
            plan = build_s3_upload_plan(
                manifest,
                inventory_path,
                bucket="safe-dataset-bucket",
                prefix="robotics/surface-v07",
                server_side_encryption="AES256",
                endpoint_url="https://objects.example.internal",
            )

            self.assertEqual(plan["mode"], "upload-only-no-delete")
            self.assertEqual(
                plan["endpoint_url"], "https://objects.example.internal"
            )
            self.assertEqual(len(plan["objects"]), 8)
            self.assertTrue(
                all(inventory["dataset_fingerprint"] in item["key"] for item in plan["objects"])
            )
            dry_run = execute_s3_upload_plan(plan, dry_run=True)
            self.assertEqual(dry_run["planned"], 8)
            self.assertEqual(dry_run["deleted"], 0)

            client = _FakeS3Client()
            first_upload = execute_s3_upload_plan(plan, client=client, dry_run=False)
            second_upload = execute_s3_upload_plan(plan, client=client, dry_run=False)
            self.assertEqual(first_upload["uploaded"], 8)
            self.assertEqual(second_upload["skipped_matching"], 8)
            self.assertEqual(second_upload["deleted"], 0)
            self.assertTrue(
                all(
                    upload[3]["ServerSideEncryption"] == "AES256"
                    for upload in client.uploads
                )
            )

            with self.assertRaisesRegex(ValueError, "HTTPS URL"):
                build_s3_upload_plan(
                    manifest,
                    inventory_path,
                    bucket="safe-dataset-bucket",
                    endpoint_url="http://insecure.example.internal",
                )

    def test_s3_execution_refuses_asset_changed_after_planning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_dataset(root)
            inventory_path = root / "inventory.json"
            build_dataset_inventory(manifest, inventory_path)
            plan = build_s3_upload_plan(
                manifest,
                inventory_path,
                bucket="safe-dataset-bucket",
            )
            changed = Path(plan["objects"][0]["local_path"])
            changed.write_bytes(b"changed-after-plan")

            with self.assertRaisesRegex(ValueError, "changed after upload planning"):
                execute_s3_upload_plan(plan, client=_FakeS3Client(), dry_run=False)

    def test_experiment_contract_rejects_dirty_code_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_dataset(root)
            inventory_path = root / "inventory.json"
            build_dataset_inventory(manifest, inventory_path)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "production-experiment-1.0",
                        "experiment_id": "fixture",
                        "provenance": {"require_clean_git": True},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "contract.json"
            dirty_state = {
                "commit": "abc123",
                "branch": "fixture",
                "dirty": True,
                "tracked_diff_sha256": "0" * 64,
                "untracked_files": ["fixture.txt"],
            }
            with patch(
                "surface_perception.reproducibility.capture_code_state",
                return_value=dirty_state,
            ):
                with self.assertRaisesRegex(ValueError, "clean Git state"):
                    build_experiment_contract(
                        config_path,
                        inventory_path,
                        output,
                        repository_root=Path(__file__).parents[1],
                    )
                contract = build_experiment_contract(
                    config_path,
                    inventory_path,
                    output,
                    repository_root=Path(__file__).parents[1],
                    allow_dirty=True,
                )
            self.assertTrue(contract["code"]["dirty"])
            self.assertEqual(contract["dataset"]["samples"], 4)


if __name__ == "__main__":
    unittest.main()
