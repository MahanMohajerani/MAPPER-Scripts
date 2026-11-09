from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_FILE = SCRIPT_DIR / "V3_user_settings.json"


def now_stamp() -> str:
    return datetime.now().strftime("%Y.%m.%d.%H.%M")


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def settings_path() -> Path:
    raw = os.environ.get("NEUROPSYCHE_SETTINGS_FILE", "").strip()
    return Path(raw) if raw else DEFAULT_SETTINGS_FILE


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def deep_update(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_settings(defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = dict(defaults or {})
    stored = load_json(settings_path(), {}) or {}
    deep_update(data, stored)
    return data


def save_settings(data: Mapping[str, Any]) -> None:
    save_json_atomic(settings_path(), dict(data))


def normalize_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_hash(data: Any, length: int = 16) -> str:
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def stage_number(script_name: str) -> Optional[int]:
    match = re.match(r"^(\d+)_", Path(script_name).name)
    return int(match.group(1)) if match else None


def stage_folder_name(script_name: str) -> str:
    number = stage_number(script_name)
    return f"stage{number}" if number is not None else Path(script_name).stem


def stage_report_root(reports_dir: Path, script_name: str) -> Path:
    return reports_dir / stage_folder_name(script_name)


def _run_sort_key(path: Path) -> tuple[float, str]:
    manifest = load_json(path / "manifest.json", {}) or {}
    recorded = next(
        (
            str(manifest[key])
            for key in (
                "finished_utc",
                "completed_utc",
                "resumed_utc",
                "started_utc",
                "created_utc",
            )
            if manifest.get(key)
        ),
        "",
    )
    if recorded:
        try:
            value = datetime.fromisoformat(recorded.replace("Z", "+00:00")).timestamp()
            return value, path.name
        except ValueError:
            pass
    try:
        stamp = path.name.split("_", 1)[0]
        return datetime.strptime(stamp, "%Y.%m.%d.%H.%M").timestamp(), path.name
    except ValueError:
        return path.stat().st_mtime, path.name


def run_dirs(reports_dir: Path, script_name: str) -> list[Path]:
    root = stage_report_root(reports_dir, script_name)
    if not root.exists():
        return []
    candidates = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    return sorted(candidates, key=_run_sort_key, reverse=True)


def manifest_complete(run_dir: Path) -> bool:
    manifest = load_json(Path(run_dir) / "manifest.json", {}) or {}
    status = str(manifest.get("status", "")).lower()
    if status:
        return status == "complete"
    return any(Path(run_dir, name).exists() for name in (
        "PREPROCESSING_COMPLETE.flag",
        "EPOCH_PREPROCESSING_COMPLETE.flag",
        "QC_COMPLETE.flag",
        "FEATURE_CACHE_COMPLETE.flag",
        "FOLDS_COMPLETE.flag",
        "NESTED_COMPLETE.flag",
        "REPORT_COMPLETE.flag",
        "INFERENCE_COMPLETE.flag",
    ))


def latest_run_dir(
    reports_dir: Path,
    script_name: str,
    *,
    complete_only: bool = False,
    ordinal_scheme: Optional[str] = None,
    required_files: Sequence[str] = (),
) -> Optional[Path]:
    for candidate in run_dirs(reports_dir, script_name):
        if complete_only and not manifest_complete(candidate):
            continue
        if required_files and not all((candidate / name).exists() for name in required_files):
            continue
        if ordinal_scheme is not None:
            manifest = load_json(candidate / "manifest.json", {}) or {}
            if str(manifest.get("ordinal_scheme", "")) != str(ordinal_scheme):
                continue
        return candidate
    return None


def latest_matching_file(reports_dir: Path, script_name: str, patterns: Sequence[str]) -> Optional[Path]:
    run = latest_run_dir(reports_dir, script_name, complete_only=True)
    if run is None:
        return None
    for pattern in patterns:
        matches = sorted(run.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def make_stage_run_dir(
    reports_dir: Path,
    script_name: str,
    resume: bool = False,
    resume_dir: Optional[Path] = None,
) -> Path:
    env_dir = os.environ.get("NEUROPSYCHE_STAGE_RUN_DIR", "").strip()
    if env_dir:
        path = Path(env_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    if resume_dir is not None:
        path = Path(resume_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    if resume:
        for candidate in run_dirs(reports_dir, script_name):
            if not manifest_complete(candidate):
                return candidate

    root = stage_report_root(reports_dir, script_name)
    root.mkdir(parents=True, exist_ok=True)
    path = root / now_stamp()
    counter = 2
    while path.exists():
        path = root / f"{now_stamp()}_{counter:02d}"
        counter += 1
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


@dataclass
class RunControl:
    control_dir: Path
    poll_seconds: float = 0.75

    @classmethod
    def from_environment(cls, fallback_dir: Optional[Path] = None) -> "RunControl":
        raw = os.environ.get("NEUROPSYCHE_CONTROL_DIR", "").strip()
        directory = Path(raw) if raw else (fallback_dir or (SCRIPT_DIR / "_control"))
        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory)

    @property
    def pause_file(self) -> Path:
        return self.control_dir / "PAUSE_REQUESTED.flag"

    @property
    def stop_file(self) -> Path:
        return self.control_dir / "STOP_REQUESTED.flag"

    def clear(self) -> None:
        for path in (self.pause_file, self.stop_file):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def stop_requested(self) -> bool:
        return self.stop_file.exists()

    def checkpoint(self, label: str = "") -> None:
        if self.stop_requested():
            raise GracefulStop(label or "Stop requested")
        announced = False
        while self.pause_file.exists():
            if not announced:
                print(f"[PAUSED] {label or 'Waiting at safe checkpoint'}", flush=True)
                announced = True
            if self.stop_requested():
                raise GracefulStop(label or "Stop requested while paused")
            time.sleep(self.poll_seconds)
        if announced:
            print(f"[RESUMED] {label or 'Continuing'}", flush=True)


class GracefulStop(RuntimeError):
    pass


class CheckpointStore:
    """Small durable set of completed work-unit keys."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed = set()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    key = line.strip()
                    if key:
                        self._completed.add(key)

    def contains(self, key: str) -> bool:
        return key in self._completed

    def add(self, key: str) -> None:
        if key in self._completed:
            return
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(key + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._completed.add(key)

    def __len__(self) -> int:
        return len(self._completed)


class JsonlStore:
    """Append-only JSONL store with optional de-duplication by a key field."""

    def __init__(self, path: Path, key_field: Optional[str] = None):
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keys = set()
        if key_field and self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if key_field in row:
                            self.keys.add(str(row[key_field]))
                    except Exception:
                        continue

    def contains(self, key: str) -> bool:
        return str(key) in self.keys

    def append(self, row: Mapping[str, Any]) -> bool:
        if self.key_field:
            key = str(row.get(self.key_field, ""))
            if key and key in self.keys:
                return False
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if self.key_field and key:
            self.keys.add(key)
        return True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_manifest(
    run_dir: Path,
    payload: Mapping[str, Any],
    filename: str = "manifest.json",
    *,
    status: Optional[str] = "complete",
) -> Path:
    path = Path(run_dir) / filename
    manifest = load_json(path, {}) or {}
    manifest.update(dict(payload))
    manifest.setdefault("created_utc", utc_iso())
    manifest.setdefault("python", sys.version)
    if status is not None:
        manifest["status"] = status
        if status == "complete":
            manifest["completed_utc"] = utc_iso()
    save_json_atomic(path, manifest)
    return path


def update_manifest(run_dir: Path, **changes: Any) -> Path:
    path = Path(run_dir) / "manifest.json"
    manifest = load_json(path, {}) or {}
    manifest.update(changes)
    manifest.setdefault("created_utc", utc_iso())
    manifest.setdefault("python", sys.version)
    save_json_atomic(path, manifest)
    return path


def append_csv_row(path: Path, row: Mapping[str, Any], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    names = list(fieldnames or row.keys())
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in names})
        f.flush()
        os.fsync(f.fileno())


def coerce_text(value: str, current: Any) -> Any:
    text = value.strip()
    if isinstance(current, bool):
        return text.lower() in {"1", "true", "yes", "y", "on"}
    if current is None:
        if text.lower() in {"", "none", "null"}:
            return None
        try:
            return json.loads(text)
        except Exception:
            return text
    if isinstance(current, int) and not isinstance(current, bool):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if isinstance(current, (list, dict, tuple, set)):
        parsed = json.loads(text)
        if isinstance(current, tuple):
            return tuple(parsed)
        if isinstance(current, set):
            return set(parsed)
        return parsed
    if isinstance(current, Path):
        return Path(text)
    return text
