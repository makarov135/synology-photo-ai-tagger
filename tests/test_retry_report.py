from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from index_photos import _failed_asset_keys
from photo_tagger.config import ProjectPaths
from photo_tagger.manifest import ManifestStore
from photo_tagger.pipeline import TaggingOptions, run_tagging


class RetryReportTests(unittest.TestCase):
    def test_reads_only_failed_asset_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "items": [
                            {"asset": "year/a"},
                            {"asset": "year/b", "error": "failed"},
                            {"asset": "year/b", "error": "again"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(_failed_asset_keys(report), ("year/b",))

    def test_pipeline_selects_only_retry_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images" / "year"
            images.mkdir(parents=True)
            (images / "a.jpg").write_bytes(b"a")
            (images / "b.jpg").write_bytes(b"b")

            stats = run_tagging(
                ProjectPaths(root / "images", root / "data"),
                TaggingOptions(scan_only=True, retry_asset_keys=("year/b",)),
                lambda *_args: None,
            )

            self.assertEqual(stats.scanned, 2)
            self.assertEqual(stats.selected, 1)

    def test_incremental_selects_only_assets_missing_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images" / "2026" / "08"
            images.mkdir(parents=True)
            (images / "old.jpg").write_bytes(b"old")
            (images / "new.jpg").write_bytes(b"new")
            paths = ProjectPaths(root / "images", root / "data")
            manifest = ManifestStore(paths.manifest)
            manifest.set("2026/08/old", {"status": "written"})
            manifest.save()

            stats = run_tagging(
                paths,
                TaggingOptions(scan_only=True, incremental=True),
                lambda *_args: None,
            )

            self.assertEqual(stats.scanned, 2)
            self.assertEqual(stats.selected, 1)


if __name__ == "__main__":
    unittest.main()
