from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from photo_tagger.config import ProjectPaths
from photo_tagger.date_repair import repair_missing_capture_dates
from photo_tagger.manifest import ManifestStore
from photo_tagger.metadata import ExifTool, create_backups, locate_exiftool, sha256_file


class DateRepairIntegrationTests(unittest.TestCase):
    def test_repairs_from_manifest_time_and_preserves_repair_backup(self) -> None:
        try:
            executable = locate_exiftool()
        except FileNotFoundError:
            self.skipTest("ExifTool 尚未安装")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            data_root = root / "data"
            source = image_root / "sample.jpg"
            source.parent.mkdir()
            Image.new("RGB", (64, 48), "navy").save(source)
            old_mtime_ns = 1_560_400_981_000_000_000
            os.utime(source, ns=(old_mtime_ns, old_mtime_ns))
            original = create_backups(
                [source], image_root, data_root / "metadata_backups" / "original"
            )[source.resolve()]
            before_sha = sha256_file(source)
            new_mtime_ns = 1_787_410_800_000_000_000
            os.utime(source, ns=(new_mtime_ns, new_mtime_ns))

            paths = ProjectPaths(image_root, data_root)
            manifest = ManifestStore(paths.manifest)
            manifest.set(
                "sample",
                {
                    "status": "written",
                    "files": {
                        "sample.jpg": {
                            "path": str(source),
                            "backup_path": str(original),
                            "before_sha256": before_sha,
                            "after_sha256": before_sha,
                            "file_mtime_before_ns": old_mtime_ns,
                            "file_mtime_after_ns": old_mtime_ns,
                        }
                    },
                },
            )
            manifest.save()

            stats = repair_missing_capture_dates(
                paths,
                write=True,
                patterns=(),
                limit=None,
                exiftool_path=executable,
                progress=lambda _current, _total, _message: None,
            )

            metadata = ExifTool(executable).read(source)
            self.assertEqual(stats.candidates, 1)
            self.assertEqual(stats.repaired, 1)
            self.assertEqual(stats.failed, 0)
            self.assertEqual(source.stat().st_mtime_ns, old_mtime_ns)
            self.assertEqual(metadata["ExifIFD:DateTimeOriginal"], "2019:06:13 12:43:01")
            self.assertTrue(Path(stats.backup_run or "").is_dir())
            repair_backup = next(Path(stats.backup_run or "").rglob("sample.jpg"))
            self.assertEqual(repair_backup.stat().st_mtime_ns, new_mtime_ns)


if __name__ == "__main__":
    unittest.main()
