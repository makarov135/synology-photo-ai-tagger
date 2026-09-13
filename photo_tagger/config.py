from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "image"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
DEFAULT_MODEL_DIR = DEFAULT_DATA_DIR / "models" / "Qwen3-VL-8B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "Qwen3VL-8B-Instruct-Q8_0.gguf"
DEFAULT_MMPROJ_FILE = "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
DEFAULT_LLAMA_SERVER = PROJECT_ROOT / "tools" / "llama_cpp" / "llama-server.exe"
DEFAULT_EXIFTOOL = PROJECT_ROOT / "tools" / "exiftool" / "exiftool.exe"
DEFAULT_FRAME_RATIOS = (0.25, 0.50, 0.75)
DEFAULT_PRIMARY_MAX_EDGE = 2048
DEFAULT_LIVE_FRAME_MAX_EDGE = 1024
DEFAULT_CONTEXT_SIZE = 12288
DEFAULT_IMAGE_MIN_TOKENS = 1024
DEFAULT_IMAGE_MAX_TOKENS = 3072
PROMPT_VERSION = "zh-photo-tags-2026-09-09-v3-balanced-verification"

STILL_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".heic", ".heif", ".dng"})
HEIF_EXTENSIONS = frozenset({".heic", ".heif"})
SKIPPED_STILL_EXTENSIONS: frozenset[str] = frozenset()
MOTION_EXTENSIONS = frozenset({".mov"})
VIDEO_EXTENSIONS = frozenset({".mov", ".mp4", ".m4v"})
JPEG_EXTENSIONS = frozenset({".jpg", ".jpeg"})
EXIF_DESCRIPTION_EXTENSIONS = frozenset({".jpg", ".jpeg", ".dng"})


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    image_dir: Path = DEFAULT_IMAGE_DIR
    data_dir: Path = DEFAULT_DATA_DIR

    @property
    def manifest(self) -> Path:
        return self.data_dir / "tag_manifest.json"

    @property
    def reports(self) -> Path:
        return self.data_dir / "reports"

    @property
    def backups(self) -> Path:
        return self.data_dir / "metadata_backups"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models" / DEFAULT_MODEL_DIR.name

    def ensure_output_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
