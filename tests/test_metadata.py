from __future__ import annotations

import tempfile
import os
import unittest
from pathlib import Path

from PIL import Image

from photo_tagger.ai import PhotoTags
from photo_tagger.metadata import ExifTool, FileTagState, create_backups, locate_exiftool


class MetadataIntegrationTests(unittest.TestCase):
    def test_missing_capture_date_is_filled_from_original_file_time(self) -> None:
        try:
            executable = locate_exiftool()
        except FileNotFoundError:
            self.skipTest("ExifTool 尚未安装")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            source = image_root / "without-date.jpg"
            source.parent.mkdir()
            Image.new("RGB", (64, 48), "navy").save(source)
            old_mtime_ns = 1_560_400_981_000_000_000
            os.utime(source, ns=(old_mtime_ns, old_mtime_ns))
            backups = create_backups([source], image_root, root / "backups" / "run")

            tool = ExifTool(executable)
            result = tool.write(
                source,
                PhotoTags("蓝色测试图像", ("蓝色", "测试图", "横向画面", "1280.00")),
                backups[source.resolve()],
                FileTagState(),
            )
            metadata = tool.read(source)

            self.assertEqual(result.capture_time_source, "original_file_mtime")
            self.assertEqual(result.file_mtime_after_ns, old_mtime_ns)
            self.assertEqual(metadata["ExifIFD:DateTimeOriginal"], "2019:06:13 12:43:01")
            self.assertIn("蓝色", metadata["XMP-dc:Subject"])
            self.assertIn("1280.00", metadata["XMP-dc:Subject"])

    def test_jpeg_write_and_backup(self) -> None:
        try:
            executable = locate_exiftool()
        except FileNotFoundError:
            self.skipTest("ExifTool 尚未安装")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            source = image_root / "sample.jpg"
            source.parent.mkdir()
            exif = Image.Exif()
            exif[36867] = "2026:09:09 12:34:56"
            Image.new("RGB", (64, 48), "navy").save(source, exif=exif)
            backups = create_backups([source], image_root, root / "backups" / "run")

            tool = ExifTool(executable)
            result = tool.write(
                source,
                PhotoTags("蓝色测试图像", ("蓝色", "测试图", "横向画面")),
                backups[source.resolve()],
                FileTagState(),
            )
            metadata = tool.read(source)
            self.assertIn("蓝色", metadata["XMP-dc:Subject"])
            self.assertIn("蓝色", metadata["IPTC:Keywords"])
            self.assertEqual(metadata["IFD0:ImageDescription"], "蓝色测试图像")
            self.assertEqual(metadata["XMP-dc:Description"], "蓝色测试图像")
            self.assertEqual(metadata["IPTC:Caption-Abstract"], "蓝色测试图像")
            self.assertNotEqual(result.before_sha256, result.after_sha256)
            self.assertTrue(Path(result.backup_path).is_file())
            self.assertEqual(result.capture_time_source, "embedded")

    def test_long_utf8_keyword_is_kept_in_xmp_without_breaking_iptc(self) -> None:
        try:
            executable = locate_exiftool()
        except FileNotFoundError:
            self.skipTest("ExifTool 尚未安装")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            source = image_root / "long-keyword.jpg"
            source.parent.mkdir()
            Image.new("RGB", (64, 48), "navy").save(source)
            backups = create_backups([source], image_root, root / "backups" / "run")
            long_keyword = "设置本地登录用户名和密码以防止未经授权的访问"

            tool = ExifTool(executable)
            tool.write(
                source,
                PhotoTags("路由器设置界面", ("路由器", "设置界面", long_keyword)),
                backups[source.resolve()],
                FileTagState(),
            )
            metadata = tool.read(source)

            self.assertIn(long_keyword, metadata["XMP-dc:Subject"])
            self.assertNotIn(long_keyword, metadata["IPTC:Keywords"])

    def test_synology_refresh_updates_only_filesystem_mtime(self) -> None:
        try:
            executable = locate_exiftool()
        except FileNotFoundError:
            self.skipTest("ExifTool 尚未安装")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            source = image_root / "sample.jpg"
            source.parent.mkdir()
            exif = Image.Exif()
            exif[36867] = "2026:09:09 12:34:56"
            Image.new("RGB", (64, 48), "navy").save(source, exif=exif)
            old_mtime_ns = 946684800_000_000_000
            os.utime(source, ns=(old_mtime_ns, old_mtime_ns))
            backups = create_backups([source], image_root, root / "backups" / "run")

            tool = ExifTool(executable)
            result = tool.write(
                source,
                PhotoTags("蓝色测试图像", ("蓝色", "测试图", "横向画面")),
                backups[source.resolve()],
                FileTagState(),
                synology_refresh=True,
            )

            self.assertEqual(result.file_mtime_before_ns, old_mtime_ns)
            self.assertNotEqual(result.file_mtime_after_ns, old_mtime_ns)
            self.assertTrue(result.synology_refresh)
            metadata = tool.read(source)
            capture_time = next(
                value
                for key, value in metadata.items()
                if key.rsplit(":", 1)[-1] == "DateTimeOriginal"
            )
            self.assertEqual(capture_time, "2026:09:09 12:34:56")

if __name__ == "__main__":
    unittest.main()
