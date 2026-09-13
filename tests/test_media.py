from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from photo_tagger.media import scan_assets


class ScanAssetsTests(unittest.TestCase):
    def test_groups_live_photo_and_duplicate_stills(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "IMG_0001.HEIC",
                "IMG_0001.JPG",
                "IMG_0001.MOV",
                "IMG_0002.HEIC",
                "IMG_0002.MOV",
                "clip.mp4",
            ):
                (root / name).write_bytes(name.encode("ascii"))
            assets = scan_assets(root)
            self.assertEqual(len(assets), 3)
            live = next(item for item in assets if item.display_name == "IMG_0001")
            self.assertTrue(live.is_live_photo)
            self.assertEqual(len(live.still_paths), 2)
            self.assertEqual(len(live.ignored_paths), 0)
            self.assertEqual(live.primary_path.suffix, ".JPG")
            heic_live = next(item for item in assets if item.display_name == "IMG_0002")
            self.assertEqual(len(heic_live.still_paths), 1)
            self.assertEqual(len(heic_live.ignored_paths), 0)
            self.assertEqual(heic_live.primary_path.suffix, ".HEIC")
            self.assertTrue(heic_live.is_live_photo)
            video = next(item for item in assets if item.display_name == "clip")
            self.assertFalse(video.still_paths)


if __name__ == "__main__":
    unittest.main()
