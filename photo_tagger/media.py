from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageOps

from .config import (
    HEIF_EXTENSIONS,
    MOTION_EXTENSIONS,
    SKIPPED_STILL_EXTENSIONS,
    STILL_EXTENSIONS,
    VIDEO_EXTENSIONS,
)


_STILL_PRIORITY = {
    ".jpg": 0,
    ".jpeg": 0,
    ".heic": 1,
    ".heif": 1,
    ".png": 2,
    ".dng": 3,
}


@dataclass(frozen=True, slots=True)
class AssetCandidate:
    """One logical photo, including duplicate stills and a Live Photo MOV."""

    key: str
    display_name: str
    primary_path: Path | None
    still_paths: tuple[Path, ...]
    motion_paths: tuple[Path, ...]
    signature: str
    ignored_paths: tuple[Path, ...] = ()

    @property
    def is_live_photo(self) -> bool:
        return bool(self.still_paths) and any(
            path.suffix.casefold() in MOTION_EXTENSIONS for path in self.motion_paths
        )


def scan_assets(
    image_dir: Path,
    scan_progress: Callable[[str, int, int], None] | None = None,
    signature_filter: Callable[[str], bool] | None = None,
) -> list[AssetCandidate]:
    """Group same-stem stills and video companions without changing any files."""

    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{image_dir}")

    supported = STILL_EXTENSIONS | VIDEO_EXTENSIONS
    media_paths: list[Path] = []
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in supported:
            continue
        media_paths.append(path)
        if scan_progress is not None and len(media_paths) % 250 == 0:
            scan_progress("遍历目录", len(media_paths), 0)

    if scan_progress is not None:
        scan_progress("遍历目录完成", len(media_paths), len(media_paths))

    groups: dict[tuple[str, str], dict[str, list[Path]]] = {}
    sorted_paths = sorted(media_paths, key=lambda item: str(item).casefold())
    for index, path in enumerate(sorted_paths, start=1):
        relative_parent = path.parent.relative_to(image_dir).as_posix()
        group_key = (relative_parent.casefold(), path.stem.casefold())
        group = groups.setdefault(group_key, {"stills": [], "motions": [], "ignored": []})
        extension = path.suffix.casefold()
        if extension in SKIPPED_STILL_EXTENSIONS:
            bucket = "ignored"
        elif extension in STILL_EXTENSIONS:
            bucket = "stills"
        else:
            bucket = "motions"
        group[bucket].append(path.resolve())
        if scan_progress is not None and (
            index == 1 or index % 500 == 0 or index == len(sorted_paths)
        ):
            scan_progress("整理媒体文件", index, len(sorted_paths))

    assets: list[AssetCandidate] = []
    sorted_groups = sorted(groups.items())
    for index, ((relative_parent, stem), grouped) in enumerate(sorted_groups, start=1):
        stills = tuple(
            grouped["stills"]
            if len(grouped["stills"]) < 2
            else sorted(grouped["stills"], key=_still_sort_key)
        )
        motions = tuple(sorted(grouped["motions"], key=lambda item: item.name.casefold()))
        ignored = tuple(sorted(grouped["ignored"], key=lambda item: item.name.casefold()))
        primary = stills[0] if stills else None
        key = f"{relative_parent}/{stem}" if relative_parent != "." else stem
        display_source = primary or (ignored[0] if ignored else motions[0])
        display_name = display_source.stem
        paths = (*stills, *motions)
        signature = (
            asset_signature(paths, image_dir)
            if signature_filter is None or signature_filter(key)
            else ""
        )
        assets.append(
            AssetCandidate(
                key=key,
                display_name=display_name,
                primary_path=primary,
                still_paths=stills,
                motion_paths=motions,
                signature=signature,
                ignored_paths=ignored,
            )
        )
        if scan_progress is not None and (
            index == 1 or index % 100 == 0 or index == len(sorted_groups)
        ):
            scan_progress("读取媒体文件状态", index, len(sorted_groups))
    return assets


def refreshed_asset(candidate: AssetCandidate, image_dir: Path) -> AssetCandidate:
    return AssetCandidate(
        key=candidate.key,
        display_name=candidate.display_name,
        primary_path=candidate.primary_path,
        still_paths=candidate.still_paths,
        motion_paths=candidate.motion_paths,
        signature=asset_signature((*candidate.still_paths, *candidate.motion_paths), image_dir),
        ignored_paths=candidate.ignored_paths,
    )


def _still_sort_key(path: Path) -> tuple[int, int, str]:
    return (_STILL_PRIORITY.get(path.suffix.casefold(), 99), -path.stat().st_size, path.name.casefold())


def asset_signature(paths: Iterable[Path], image_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item).casefold()):
        stat = path.stat()
        relative = path.resolve().relative_to(image_dir.resolve()).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def load_still_image(path: Path, max_edge: int = 1024) -> Image.Image:
    """Decode a supported still image and return an oriented RGB preview."""

    extension = path.suffix.casefold()
    if extension in HEIF_EXTENSIONS:
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener(thumbnails=False)
        except ImportError as exc:
            raise RuntimeError(
                "缺少 HEIC 解码组件；请运行 pip install -r requirements.txt"
            ) from exc
    if extension == ".dng":
        raw_error: Exception | None = None
        try:
            import rawpy

            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            image = Image.fromarray(rgb, mode="RGB")
        except Exception as exc:
            raw_error = exc
            try:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
            except Exception as pillow_error:
                raise RuntimeError(
                    f"DNG 解码失败（RAW：{raw_error}；Pillow：{pillow_error}）"
                ) from pillow_error
    else:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image


def locate_ffmpeg(explicit: str | Path | None, executable: str) -> str | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return str(path) if path.is_file() else None
    discovered = shutil.which(executable)
    if discovered:
        return discovered
    windows_default = Path("C:/ffmpeg/bin") / f"{executable}.exe"
    return str(windows_default) if windows_default.is_file() else None


def extract_video_frames(
    path: Path,
    ratios: Sequence[float],
    *,
    max_edge: int = 1024,
    ffmpeg_path: str | Path | None = None,
    ffprobe_path: str | Path | None = None,
) -> list[tuple[float, Image.Image]]:
    """Extract representative Live Photo frames; the MOV remains read-only."""

    ffmpeg = locate_ffmpeg(ffmpeg_path, "ffmpeg")
    ffprobe = locate_ffmpeg(ffprobe_path, "ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("读取实况照片需要 FFmpeg（ffmpeg 和 ffprobe）")

    duration = _probe_duration(path, ffprobe)
    if duration <= 0:
        raise RuntimeError(f"无法读取视频时长：{path}")

    frames: list[tuple[float, Image.Image]] = []
    for raw_ratio in ratios:
        ratio = min(0.98, max(0.02, float(raw_ratio)))
        timestamp = duration * ratio
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.4f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={max_edge}:{max_edge}:force_original_aspect_ratio=decrease",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        completed = subprocess.run(command, capture_output=True, check=False, timeout=45)
        if completed.returncode != 0 or not completed.stdout:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg 抽帧失败：{path.name}；{message}")
        with Image.open(io.BytesIO(completed.stdout)) as source:
            frames.append((ratio, source.convert("RGB").copy()))
    return frames


def _probe_duration(path: Path, ffprobe: str) -> float:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        return 0.0
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return 0.0
