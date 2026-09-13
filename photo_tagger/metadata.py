from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .ai import PhotoTags
from .config import (
    DEFAULT_EXIFTOOL,
    EXIF_DESCRIPTION_EXTENSIONS,
    JPEG_EXTENSIONS,
    STILL_EXTENSIONS,
)


class MetadataError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FileTagState:
    ai_keywords: tuple[str, ...] = ()
    ai_description: str | None = None
    preserved_keywords: tuple[str, ...] = ()
    preserved_description: str | None = None

    @classmethod
    def from_dict(cls, value: object) -> "FileTagState":
        if not isinstance(value, dict):
            return cls()
        return cls(
            ai_keywords=tuple(str(item) for item in value.get("ai_keywords", [])),
            ai_description=value.get("ai_description"),
            preserved_keywords=tuple(str(item) for item in value.get("preserved_keywords", [])),
            preserved_description=value.get("preserved_description"),
        )


@dataclass(frozen=True, slots=True)
class MetadataWriteResult:
    path: str
    backup_path: str
    before_sha256: str
    after_sha256: str
    written_keywords: tuple[str, ...]
    written_description: str | None
    synology_refresh: bool
    file_mtime_before_ns: int
    file_mtime_after_ns: int
    capture_time_source: str
    state: FileTagState

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["written_keywords"] = list(self.written_keywords)
        result["state"]["ai_keywords"] = list(self.state.ai_keywords)
        result["state"]["preserved_keywords"] = list(self.state.preserved_keywords)
        return result


def locate_exiftool(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(DEFAULT_EXIFTOOL)
    discovered = shutil.which("exiftool")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "没有找到 ExifTool。请先运行 setup_tools.py，或用 --exiftool 指定路径。"
    )


class ExifTool:
    def __init__(self, executable: str | Path | None = None) -> None:
        self.executable = locate_exiftool(executable)

    def version(self) -> str:
        return self._run(["-ver"]).stdout.strip()

    def has_embedded_capture_time(self, path: Path) -> bool:
        return _has_embedded_capture_time(self.read(path))

    def read(self, path: Path) -> dict[str, Any]:
        completed = self._run(
            [
                "-json",
                "-G1",
                "-s",
                "-api",
                "StructFormat=JSONQ",
                "-charset",
                "EXIF=UTF8",
                "-charset",
                "IPTC=UTF8",
                str(path),
            ]
        )
        try:
            values = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise MetadataError(f"ExifTool 返回了无效 JSON：{path.name}") from exc
        if not values or not isinstance(values[0], dict):
            raise MetadataError(f"ExifTool 没有读取到元数据：{path.name}")
        return values[0]

    def read_many(self, paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
        requested = [path.resolve() for path in paths]
        if not requested:
            return {}
        completed = self._run(
            [
                "-json",
                "-G1",
                "-s",
                "-api",
                "StructFormat=JSONQ",
                "-charset",
                "EXIF=UTF8",
                "-charset",
                "IPTC=UTF8",
                *(str(path) for path in requested),
            ]
        )
        try:
            values = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise MetadataError("ExifTool 批量读取返回了无效 JSON") from exc
        result: dict[str, dict[str, Any]] = {}
        for value in values if isinstance(values, list) else []:
            if not isinstance(value, dict) or not value.get("SourceFile"):
                continue
            key = str(Path(str(value["SourceFile"])).resolve()).casefold()
            result[key] = value
        return result

    def write(
        self,
        path: Path,
        tags: PhotoTags,
        backup_path: Path,
        previous: FileTagState,
        *,
        synology_refresh: bool = False,
    ) -> MetadataWriteResult:
        before = self.read(path)
        mtime_before_ns = path.stat().st_mtime_ns
        if (
            synology_refresh
            and not _has_embedded_capture_time(before)
            and not extension_supports_date_fallback(path.suffix)
        ):
            raise MetadataError(
                f"为避免改变相册顺序，拒绝刷新缺少内嵌拍摄时间的文件：{path.name}"
            )
        before_sha = sha256_file(path)
        existing_keywords = _search_keywords(before)
        existing_description = _search_description(before)
        preserved_keywords = _updated_preserved_keywords(existing_keywords, previous)
        desired_keywords = tuple(_deduplicate((*preserved_keywords, *tags.keywords)))
        preserved_description, desired_description = _desired_description(
            existing_description, tags.description, previous
        )
        critical_before = _critical_metadata(before)
        extension = path.suffix.casefold()
        capture_time_source = "embedded"
        fallback_capture_time: datetime | None = None
        if not _has_embedded_capture_time(before) and extension_supports_date_fallback(
            path.suffix
        ):
            fallback_capture_time = datetime.fromtimestamp(
                mtime_before_ns / 1_000_000_000
            ).astimezone()
            capture_time_source = "original_file_mtime"

        command = [
            "-m",
            "-charset",
            "EXIF=UTF8",
            "-charset",
            "IPTC=UTF8",
        ]
        if not synology_refresh:
            command.append("-P")
        command.extend(["-overwrite_original_in_place", "-XMP-dc:Subject="])
        command.extend(f"-XMP-dc:Subject+={keyword}" for keyword in desired_keywords)
        command.append(f"-XMP-dc:Description={desired_description}")

        if extension in JPEG_EXTENSIONS:
            # IPTC-IIM Keywords are limited to 64 encoded bytes per value. Keep the
            # complete searchable set in XMP and mirror only compatible values to
            # IPTC so a long Chinese OCR phrase cannot make the whole write fail.
            iptc_keywords = tuple(
                keyword for keyword in desired_keywords if _is_iptc_keyword(keyword)
            )
            command.extend(["-IPTC:Keywords=", "-IPTC:CodedCharacterSet=UTF8"])
            command.extend(f"-IPTC:Keywords+={keyword}" for keyword in iptc_keywords)
            command.append(f"-IPTC:Caption-Abstract={desired_description}")
        if extension in EXIF_DESCRIPTION_EXTENSIONS:
            command.append(f"-EXIF:ImageDescription={desired_description}")
        if fallback_capture_time is not None:
            command.extend(_capture_time_arguments(extension, fallback_capture_time))
        command.append(str(path))

        try:
            completed = self._run(command)
            if "image files updated" not in completed.stdout.casefold():
                raise MetadataError(
                    f"ExifTool 未确认写入成功：{path.name}；{completed.stdout.strip()}"
                )
            after = self.read(path)
            _verify_written(path, after, desired_keywords, desired_description)
            critical_after = _critical_metadata(after)
            comparable_after = (
                {key: value for key, value in critical_after.items() if key in critical_before}
                if fallback_capture_time is not None
                else critical_after
            )
            if critical_before != comparable_after:
                differences = _dictionary_difference(critical_before, comparable_after)
                raise MetadataError(f"关键原始元数据发生变化：{differences}")
            if fallback_capture_time is not None:
                _verify_capture_time(path, after, fallback_capture_time)
        except Exception:
            shutil.copy2(backup_path, path)
            if sha256_file(path) != before_sha:
                raise MetadataError(f"写入失败，且自动恢复未能还原：{path}")
            raise

        state = FileTagState(
            ai_keywords=tags.keywords,
            ai_description=tags.description if desired_description == tags.description else None,
            preserved_keywords=preserved_keywords,
            preserved_description=preserved_description,
        )
        return MetadataWriteResult(
            path=str(path),
            backup_path=str(backup_path),
            before_sha256=before_sha,
            after_sha256=sha256_file(path),
            written_keywords=desired_keywords,
            written_description=desired_description,
            synology_refresh=synology_refresh,
            file_mtime_before_ns=mtime_before_ns,
            file_mtime_after_ns=path.stat().st_mtime_ns,
            capture_time_source=capture_time_source,
            state=state,
        )

    def ensure_capture_time(
        self,
        path: Path,
        original_mtime_ns: int,
        backup_path: Path,
    ) -> str | None:
        """Embed the former filesystem date without removing current AI metadata."""
        before = self.read(path)
        if _has_embedded_capture_time(before):
            return None
        if not extension_supports_date_fallback(path.suffix):
            return None
        before_sha = sha256_file(path)
        before_keywords = _search_keywords(before)
        before_description = _search_description(before)
        critical_before = _critical_metadata(before)
        capture_time = datetime.fromtimestamp(
            original_mtime_ns / 1_000_000_000
        ).astimezone()
        command = [
            "-m",
            "-charset",
            "EXIF=UTF8",
            "-charset",
            "IPTC=UTF8",
            "-P",
            "-overwrite_original_in_place",
            *_capture_time_arguments(path.suffix.casefold(), capture_time),
            str(path),
        ]
        try:
            completed = self._run(command)
            if "image files updated" not in completed.stdout.casefold():
                raise MetadataError(
                    f"ExifTool 未确认日期修复成功：{path.name}；{completed.stdout.strip()}"
                )
            after = self.read(path)
            _verify_capture_time(path, after, capture_time)
            if before_keywords != _search_keywords(after):
                raise MetadataError(f"日期修复改变了关键词：{path.name}")
            if before_description != _search_description(after):
                raise MetadataError(f"日期修复改变了描述：{path.name}")
            critical_after = _critical_metadata(after)
            comparable_after = {
                key: value for key, value in critical_after.items() if key in critical_before
            }
            if critical_before != comparable_after:
                differences = _dictionary_difference(critical_before, comparable_after)
                raise MetadataError(f"日期修复改变了其他关键元数据：{differences}")
            current_stat = path.stat()
            os.utime(path, ns=(current_stat.st_atime_ns, original_mtime_ns))
        except Exception:
            shutil.copy2(backup_path, path)
            if sha256_file(path) != before_sha:
                raise MetadataError(f"日期修复失败，且自动恢复未能还原：{path}")
            raise
        return capture_time.isoformat(timespec="seconds")

    def _run(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        # Windows recodes command-line arguments through the active ANSI code
        # page.  A UTF-8 -@ argument file is ExifTool's documented way to keep
        # Chinese values and Unicode filenames lossless.
        argument_file: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                suffix=".args",
                delete=False,
            ) as handle:
                argument_file = Path(handle.name)
                for argument in arguments:
                    if "\n" in argument or "\r" in argument:
                        raise MetadataError("ExifTool 参数中不能包含换行符")
                    handle.write(argument + "\n")
            environment = os.environ.copy()
            if os.name == "nt":
                environment.pop("LC_ALL", None)
                environment.pop("LC_CTYPE", None)
                environment.pop("LANG", None)
            completed = subprocess.run(
                [
                    str(self.executable),
                    "-charset",
                    "UTF8",
                    "-charset",
                    "filename=UTF8",
                    "-@",
                    str(argument_file),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=180,
                env=environment,
            )
        finally:
            if argument_file is not None:
                argument_file.unlink(missing_ok=True)
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise MetadataError(f"ExifTool 执行失败（{completed.returncode}）：{details}")
        return completed


def create_backups(
    paths: Iterable[Path], image_root: Path, backup_run_dir: Path
) -> dict[Path, Path]:
    image_root = image_root.resolve()
    backup_run_dir = backup_run_dir.resolve()
    result: dict[Path, Path] = {}
    for source in paths:
        source = source.resolve()
        if not source.is_relative_to(image_root):
            raise MetadataError(f"拒绝备份图片目录以外的文件：{source}")
        if source.suffix.casefold() not in STILL_EXTENSIONS:
            raise MetadataError(f"拒绝修改非静态照片：{source}")
        destination = (backup_run_dir / source.relative_to(image_root)).resolve()
        if not destination.is_relative_to(backup_run_dir):
            raise MetadataError(f"非法备份路径：{destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise MetadataError(f"备份文件已存在，拒绝覆盖：{destination}")
        shutil.copy2(source, destination)
        if sha256_file(source) != sha256_file(destination):
            raise MetadataError(f"备份校验失败：{source.name}")
        result[source] = destination
    return result


def restore_files(backups: dict[Path, Path]) -> None:
    failures: list[str] = []
    for target, backup in backups.items():
        try:
            shutil.copy2(backup, target)
            if sha256_file(target) != sha256_file(backup):
                failures.append(str(target))
        except Exception:
            failures.append(str(target))
    if failures:
        raise MetadataError("以下文件恢复失败：" + "，".join(failures))


def restore_backup_run(backup_run_dir: Path, image_root: Path) -> int:
    backup_run_dir = backup_run_dir.resolve()
    image_root = image_root.resolve()
    if not backup_run_dir.is_dir():
        raise FileNotFoundError(f"备份批次不存在：{backup_run_dir}")
    restored = 0
    for backup in sorted(backup_run_dir.rglob("*")):
        if not backup.is_file() or backup.suffix.casefold() not in STILL_EXTENSIONS:
            continue
        target = (image_root / backup.relative_to(backup_run_dir)).resolve()
        if not target.is_relative_to(image_root):
            raise MetadataError(f"非法恢复目标：{target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, target)
        if sha256_file(target) != sha256_file(backup):
            raise MetadataError(f"恢复后校验失败：{target}")
        restored += 1
    return restored


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values_for(metadata: dict[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if value is not None and str(value).strip():
        return (str(value).strip(),)
    return ()


def _search_keywords(metadata: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _deduplicate(
            (*_values_for(metadata, "XMP-dc:Subject"), *_values_for(metadata, "IPTC:Keywords"))
        )
    )


def _search_description(metadata: dict[str, Any]) -> str | None:
    for key in (
        "IFD0:ImageDescription",
        "ExifIFD:ImageDescription",
        "XMP-dc:Description",
        "IPTC:Caption-Abstract",
    ):
        value = _description_value(metadata, key)
        if value:
            return value
    return None


def _description_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, dict):
        value = value.get("x-default") or next(iter(value.values()), None)
    if value is not None and str(value).strip():
        return str(value).strip()
    return None


def _updated_preserved_keywords(
    existing: tuple[str, ...], previous: FileTagState
) -> tuple[str, ...]:
    if not previous.ai_keywords and not previous.preserved_keywords:
        return existing
    known = {item.casefold() for item in (*previous.ai_keywords, *previous.preserved_keywords)}
    newly_added_by_user = tuple(item for item in existing if item.casefold() not in known)
    return tuple(_deduplicate((*previous.preserved_keywords, *newly_added_by_user)))


def _desired_description(
    existing: str | None, generated: str, previous: FileTagState
) -> tuple[str | None, str]:
    if previous.preserved_description:
        if existing and existing not in {previous.preserved_description, previous.ai_description}:
            return existing, existing
        return previous.preserved_description, previous.preserved_description
    if previous.ai_description and existing and existing != previous.ai_description:
        return existing, existing
    if existing and not previous.ai_description:
        return existing, existing
    return None, generated


_CRITICAL_NAMES = {
    "AssetIdentifier",
    "ContentIdentifier",
    "MediaGroupUUID",
    "DateTimeOriginal",
    "CreateDate",
    "DateCreated",
    "OffsetTimeOriginal",
    "GPSLatitude",
    "GPSLongitude",
    "GPSLatitudeRef",
    "GPSLongitudeRef",
    "Orientation",
    "Make",
    "Model",
    "LensModel",
    "ImageWidth",
    "ImageHeight",
}


def _critical_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key.rsplit(":", 1)[-1] in _CRITICAL_NAMES
        and not key.startswith(("File:", "System:", "Composite:"))
    }


def _has_embedded_capture_time(metadata: dict[str, Any]) -> bool:
    return any(
        value
        and key.rsplit(":", 1)[-1] in {"DateTimeOriginal", "CreateDate", "DateCreated"}
        and not key.startswith(("File:", "System:", "Composite:"))
        for key, value in metadata.items()
    )


def extension_supports_date_fallback(suffix: str) -> bool:
    return suffix.casefold() in {*JPEG_EXTENSIONS, ".png"}


def metadata_has_embedded_capture_time(metadata: dict[str, Any]) -> bool:
    return _has_embedded_capture_time(metadata)


def _capture_time_arguments(extension: str, value: datetime) -> list[str]:
    exif_value = value.strftime("%Y:%m:%d %H:%M:%S")
    offset = value.strftime("%z")
    xmp_offset = f"{offset[:3]}:{offset[3:]}" if offset else ""
    xmp_value = value.strftime("%Y-%m-%dT%H:%M:%S") + xmp_offset
    arguments = [
        f"-XMP-exif:DateTimeOriginal={xmp_value}",
        f"-XMP-xmp:CreateDate={xmp_value}",
        f"-XMP-photoshop:DateCreated={xmp_value}",
    ]
    if extension.casefold() in JPEG_EXTENSIONS:
        arguments.extend(
            [
                f"-EXIF:DateTimeOriginal={exif_value}",
                f"-EXIF:CreateDate={exif_value}",
                f"-IPTC:DateCreated={value.strftime('%Y:%m:%d')}",
                f"-IPTC:TimeCreated={value.strftime('%H:%M:%S')}{xmp_offset}",
            ]
        )
    return arguments


def _verify_capture_time(
    path: Path, metadata: dict[str, Any], expected: datetime
) -> None:
    expected_prefixes = {
        expected.strftime("%Y:%m:%d %H:%M:%S"),
        expected.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    values = [
        str(value)
        for key, value in metadata.items()
        if key.rsplit(":", 1)[-1] == "DateTimeOriginal" and value
    ]
    if not values or not any(
        any(value.startswith(prefix) for prefix in expected_prefixes) for value in values
    ):
        raise MetadataError(f"回读内嵌拍摄日期失败：{path.name}")


def _verify_written(
    path: Path,
    metadata: dict[str, Any],
    expected_keywords: tuple[str, ...],
    expected_description: str,
) -> None:
    expected = {item.casefold() for item in expected_keywords}
    xmp = {item.casefold() for item in _values_for(metadata, "XMP-dc:Subject")}
    missing_xmp = [item for item in expected_keywords if item.casefold() not in xmp]
    if missing_xmp:
        raise MetadataError(f"回读 XMP Subject 缺少关键词：{path.name}：{missing_xmp}")
    if path.suffix.casefold() in JPEG_EXTENSIONS:
        iptc = {item.casefold() for item in _values_for(metadata, "IPTC:Keywords")}
        iptc_expected = {
            item.casefold() for item in expected_keywords if _is_iptc_keyword(item)
        }
        if not iptc_expected.issubset(iptc):
            raise MetadataError(f"回读 IPTC Keywords 不完整：{path.name}")
        iptc_description = _description_value(metadata, "IPTC:Caption-Abstract")
        if iptc_description != expected_description:
            raise MetadataError(
                f"回读 IPTC Caption-Abstract 不一致：{path.name}；"
                f"期望 {expected_description!r}，实际 {iptc_description!r}"
            )
    actual_description = _search_description(metadata)
    if actual_description != expected_description:
        raise MetadataError(
            f"回读描述不一致：{path.name}；期望 {expected_description!r}，实际 {actual_description!r}"
        )


def _is_iptc_keyword(value: str) -> bool:
    return len(value.encode("utf-8")) <= 64


def _dictionary_difference(before: dict[str, Any], after: dict[str, Any]) -> str:
    keys = sorted(set(before) | set(after))
    parts = [
        f"{key}: {before.get(key)!r} -> {after.get(key)!r}"
        for key in keys
        if before.get(key) != after.get(key)
    ]
    return "; ".join(parts[:8])


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result
