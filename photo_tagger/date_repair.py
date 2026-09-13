from __future__ import annotations

import fnmatch
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from .config import ProjectPaths
from .manifest import ManifestStore
from .media import refreshed_asset, scan_assets
from .metadata import (
    ExifTool,
    create_backups,
    extension_supports_date_fallback,
    metadata_has_embedded_capture_time,
    sha256_file,
)


Progress = Callable[[int, int, str], None]


@dataclass(slots=True)
class DateRepairStats:
    scanned_files: int = 0
    candidates: int = 0
    repaired: int = 0
    failed: int = 0
    candidate_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    failure_details: list[str] = field(default_factory=list)
    backup_run: str | None = None
    report_path: str | None = None


def repair_missing_capture_dates(
    paths: ProjectPaths,
    *,
    write: bool,
    patterns: Sequence[str],
    limit: int | None,
    exiftool_path: Path | None,
    progress: Progress,
) -> DateRepairStats:
    """Repair files whose original timeline depended only on filesystem mtime."""
    paths.ensure_output_dirs()
    manifest = ManifestStore(paths.manifest)
    tool = ExifTool(exiftool_path)
    assets = (
        {asset.key: asset for asset in scan_assets(paths.image_dir)} if write else {}
    )
    root = paths.image_dir.resolve()
    stats = DateRepairStats()
    potential: list[tuple[str, str, Path, Path, int]] = []
    candidates: list[tuple[str, str, Path, Path, int]] = []

    for asset_key, record in manifest.assets.items():
        raw_files = record.get("files")
        if record.get("status") != "written" or not isinstance(raw_files, dict):
            continue
        for relative, raw_entry in raw_files.items():
            if not isinstance(raw_entry, dict):
                continue
            target = Path(str(raw_entry.get("path") or "")).resolve()
            backup = Path(str(raw_entry.get("backup_path") or "")).resolve()
            original_mtime_ns = int(raw_entry.get("file_mtime_before_ns") or 0)
            if (
                original_mtime_ns <= 0
                or not extension_supports_date_fallback(target.suffix)
                or not target.is_relative_to(root)
                or not _matches(asset_key, str(relative), patterns)
            ):
                continue
            stats.scanned_files += 1
            try:
                if not target.is_file() or not backup.is_file():
                    continue
                if abs(target.stat().st_mtime_ns - original_mtime_ns) <= 2_000_000_000:
                    continue
                if abs(backup.stat().st_mtime_ns - original_mtime_ns) > 2_000_000_000:
                    raise RuntimeError("原始备份时间与清单不一致，拒绝自动推断拍摄日期")
                potential.append(
                    (asset_key, str(relative), target, backup, original_mtime_ns)
                )
            except Exception as exc:
                stats.failed += 1
                stats.failed_files.append(str(relative))
                stats.failure_details.append(str(exc))

    backup_metadata = tool.read_many(item[3] for item in potential)
    for item in potential:
        metadata = backup_metadata.get(str(item[3].resolve()).casefold())
        if metadata is None:
            stats.failed += 1
            stats.failed_files.append(item[1])
            stats.failure_details.append("无法读取原始备份元数据")
            continue
        if not metadata_has_embedded_capture_time(metadata):
            candidates.append(item)

    candidates.sort(key=lambda item: item[1].casefold())
    if limit is not None:
        candidates = candidates[: max(0, limit)]
    stats.candidates = len(candidates)
    stats.candidate_files = [relative for _, relative, *_ in candidates]
    progress(
        0,
        len(candidates),
        f"日期检查完成：发现 {len(candidates)} 个文件缺少内嵌日期，且文件时间已偏离写入前记录",
    )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_run = paths.backups / f"{run_id}-date-repair"
    if write:
        for index, (asset_key, relative, target, _original_backup, original_mtime_ns) in enumerate(
            candidates, start=1
        ):
            try:
                current_backup = create_backups([target], paths.image_dir, backup_run)[
                    target.resolve()
                ]
                capture_time = tool.ensure_capture_time(
                    target, original_mtime_ns, current_backup
                )
                record = manifest.get(asset_key) or {}
                raw_files = record.get("files")
                files = dict(raw_files) if isinstance(raw_files, dict) else {}
                raw_entry = files.get(relative)
                entry = dict(raw_entry) if isinstance(raw_entry, dict) else {}
                entry.update(
                    {
                        "capture_time_source": "original_file_mtime",
                        "capture_time_added": capture_time,
                        "capture_time_backup_path": str(current_backup),
                        "after_sha256": sha256_file(target),
                        "file_mtime_after_ns": target.stat().st_mtime_ns,
                    }
                )
                files[relative] = entry
                updated = {
                    **record,
                    "files": files,
                    "date_repaired_at": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                }
                asset = assets.get(asset_key)
                if asset is not None:
                    updated["signature"] = refreshed_asset(asset, paths.image_dir).signature
                manifest.set(asset_key, updated)
                manifest.save()
                stats.repaired += 1
                stats.backup_run = str(backup_run)
                progress(
                    index,
                    len(candidates),
                    f"{relative}：已补入拍摄日期并恢复原文件时间",
                )
            except Exception as exc:
                stats.failed += 1
                stats.failed_files.append(relative)
                stats.failure_details.append(str(exc))
                progress(index, len(candidates), f"{relative}：日期修复失败：{exc}")

    report_path = paths.reports / f"{run_id}-date-repair.json"
    report_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "write": write,
                "stats": asdict(stats),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stats.report_path = str(report_path)
    return stats


def _matches(asset_key: str, relative: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    values = (asset_key.casefold(), relative.casefold(), Path(relative).name.casefold())
    return any(
        fnmatch.fnmatchcase(value, pattern.casefold())
        for pattern in patterns
        for value in values
    )
