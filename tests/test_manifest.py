from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from photo_tagger.manifest import ManifestStore


class ManifestTests(unittest.TestCase):
    def test_atomic_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            store = ManifestStore(path)
            store.set("img_1", {"tags": {"keywords": ["海边"]}})
            store.save()
            loaded = ManifestStore(path)
            self.assertEqual(loaded.get("img_1")["tags"]["keywords"], ["海边"])
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
