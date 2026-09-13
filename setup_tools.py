from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
TOOLS_DIR = PROJECT_ROOT / "tools"
EXIFTOOL_VERSION = "13.59"
EXIFTOOL_ARCHIVE = f"exiftool-{EXIFTOOL_VERSION}_64.zip"
EXIFTOOL_URL = f"https://sourceforge.net/projects/exiftool/files/{EXIFTOOL_ARCHIVE}/download"
EXIFTOOL_CHECKSUMS_URL = (
    f"https://sourceforge.net/projects/exiftool/files/checksums-{EXIFTOOL_VERSION}.txt/download"
)
LLAMA_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description="下载并校验 ExifTool 与 llama.cpp CUDA 工具")
    parser.add_argument("--force", action="store_true", help="即使已经存在也重新安装")
    args = parser.parse_args()

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="photo-tagger-tools-", dir=PROJECT_ROOT) as temporary:
        temp_dir = Path(temporary)
        install_exiftool(temp_dir, force=args.force)
        install_llama_cpp(temp_dir, force=args.force)
    print("工具准备完成。模型会在首次运行 index_photos.py 时单独下载。")
    return 0


def install_exiftool(temp_dir: Path, *, force: bool) -> None:
    target = TOOLS_DIR / "exiftool"
    executable = target / "exiftool.exe"
    if executable.is_file() and not force:
        version = _run_version(executable)
        if version == EXIFTOOL_VERSION:
            print(f"ExifTool {version} 已存在，跳过。")
            return

    checksum_text = _download_text(EXIFTOOL_CHECKSUMS_URL)
    expected = _checksum_for(checksum_text, EXIFTOOL_ARCHIVE)
    archive = temp_dir / EXIFTOOL_ARCHIVE
    _download_file(EXIFTOOL_URL, archive)
    _verify_sha256(archive, expected)
    extracted = temp_dir / "exiftool"
    _extract_zip(archive, extracted)
    source_executable = next(extracted.rglob("exiftool(-k).exe"), None)
    if source_executable is None:
        source_executable = next(extracted.rglob("exiftool.exe"), None)
    source_files = next(
        (path for path in extracted.rglob("exiftool_files") if path.is_dir()), None
    )
    if source_executable is None or source_files is None:
        raise RuntimeError("ExifTool 压缩包结构与预期不符")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_executable, executable)
    shutil.copytree(source_files, target / "exiftool_files", dirs_exist_ok=True)
    version = _run_version(executable)
    if version != EXIFTOOL_VERSION:
        raise RuntimeError(f"ExifTool 版本校验失败：期望 {EXIFTOOL_VERSION}，实际 {version}")
    print(f"已安装并校验 ExifTool {version}。")


def install_llama_cpp(temp_dir: Path, *, force: bool) -> None:
    target = TOOLS_DIR / "llama_cpp"
    executable = target / "llama-server.exe"
    if executable.is_file() and not force:
        version = _run_version(executable)
        if version:
            print(f"llama.cpp 已存在（{version.splitlines()[0]}），跳过。")
            return

    releases = json.loads(_download_text(LLAMA_RELEASE_API))
    if not isinstance(releases, list):
        raise RuntimeError("GitHub 发布接口返回格式异常")
    release, binary, runtime = _find_llama_release(releases)
    print(f"选用 llama.cpp 官方发布：{release.get('tag_name', 'unknown')}")
    target.mkdir(parents=True, exist_ok=True)
    for label, asset in (("llama.cpp", binary), ("CUDA 运行库", runtime)):
        archive = temp_dir / str(asset["name"])
        _download_file(str(asset["browser_download_url"]), archive)
        digest = str(asset.get("digest") or "")
        if not digest.startswith("sha256:"):
            raise RuntimeError(f"GitHub 发布页没有提供 {asset['name']} 的 SHA-256")
        _verify_sha256(archive, digest.removeprefix("sha256:"))
        extracted = temp_dir / ("llama" if label == "llama.cpp" else "cudart")
        _extract_zip(archive, extracted)
        source_root = _single_content_root(extracted)
        _copy_tree_contents(source_root, target)
        print(f"已安装并校验 {label}。")
    if not executable.is_file():
        raise RuntimeError("llama.cpp 压缩包中没有 llama-server.exe")
    print(_run_version(executable).splitlines()[0])


def _download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "photo-tagger-setup/2.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def _download_file(url: str, destination: Path) -> None:
    print(f"下载：{destination.name}")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        completed = subprocess.run(
            [
                curl,
                "--location",
                "--fail",
                "--retry",
                "8",
                "--retry-all-errors",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "60",
                "--continue-at",
                "-",
                "--output",
                str(destination),
                url,
            ],
            check=False,
            timeout=1800,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"curl 下载失败（{completed.returncode}）：{destination.name}")
        return

    last_error: Exception | None = None
    for attempt in range(1, 9):
        existing = destination.stat().st_size if destination.exists() else 0
        headers = {"User-Agent": "photo-tagger-setup/2.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                partial = response.status == 206 and existing > 0
                if not partial:
                    existing = 0
                content_range = response.headers.get("Content-Range", "")
                range_match = re.search(r"/(\d+)$", content_range)
                total = (
                    int(range_match.group(1))
                    if range_match
                    else existing + int(response.headers.get("Content-Length") or 0)
                )
                mode = "ab" if partial else "wb"
                downloaded = existing
                next_report = ((downloaded // (64 * 1024 * 1024)) + 1) * 64 * 1024 * 1024
                with destination.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            suffix = f"/{total / 1024**2:.0f} MB" if total else " MB"
                            print(f"  {downloaded / 1024**2:.0f}{suffix}")
                            next_report += 64 * 1024 * 1024
                if total and destination.stat().st_size < total:
                    raise OSError(
                        f"下载不完整：{destination.stat().st_size} / {total} 字节"
                    )
                return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == 8:
                break
            delay = min(20, attempt * 2)
            current = destination.stat().st_size if destination.exists() else 0
            print(
                f"  连接中断，{delay} 秒后进行第 {attempt + 1}/8 次尝试"
                f"（已保存 {current / 1024**2:.1f} MB）：{exc}"
            )
            time.sleep(delay)
    raise RuntimeError(f"多次重试后仍无法下载 {destination.name}：{last_error}")


def _checksum_for(text: str, filename: str) -> str:
    escaped = re.escape(filename)
    patterns = (
        rf"(?im)^([0-9a-f]{{64}})\s+\*?{escaped}\s*$",
        rf"(?im)^SHA256\s*\({escaped}\)\s*=\s*([0-9a-f]{{64}})\s*$",
        rf"(?im)^SHA2-256\s*\({escaped}\)\s*=\s*([0-9a-f]{{64}})\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).lower()
    raise RuntimeError(f"官方校验文件中没有找到 {filename}")


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.casefold() != expected.casefold():
        raise RuntimeError(f"SHA-256 校验失败：{path.name}\n期望 {expected}\n实际 {actual}")


def _extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"压缩包包含非法路径：{member.filename}")
        source.extractall(root)


def _single_content_root(directory: Path) -> Path:
    entries = list(directory.iterdir())
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else directory


def _copy_tree_contents(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _find_llama_release(
    releases: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    binary_pattern = re.compile(r"^llama-b\d+-bin-win-cuda-12\.4-x64\.zip$")
    runtime_pattern = re.compile(r"^cudart-llama-bin-win-cuda-12\.4-x64\.zip$")
    for release in releases:
        if release.get("draft"):
            continue
        assets = release.get("assets", [])
        binary = next(
            (asset for asset in assets if binary_pattern.match(str(asset.get("name", "")))),
            None,
        )
        runtime = next(
            (asset for asset in assets if runtime_pattern.match(str(asset.get("name", "")))),
            None,
        )
        if binary is not None and runtime is not None:
            return release, binary, runtime
    raise RuntimeError("最近 30 个 llama.cpp 官方发布中没有完整的 Windows CUDA 12.4 构建")


def _run_version(executable: Path) -> str:
    try:
        completed = subprocess.run(
            [str(executable), "--version" if executable.name.startswith("llama") else "-ver"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
    except OSError:
        return ""
    return (completed.stdout or completed.stderr).strip()


if __name__ == "__main__":
    raise SystemExit(main())
