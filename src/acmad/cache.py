"""Content-hash keys and atomic cache directories (docs/TRAFFIC.md §11.2); shared by traffic/ and sim/models/."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

MANIFEST = "manifest.json"
_HASH_MEMO = ".file_hashes.json"


def key(obj: Any) -> str:
    """16-hex-digit SHA-256 of a JSON-serialisable object (sorted keys)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_sha256(path: str | Path, memo_dir: str | Path | None = None, chunk: int = 1 << 24) -> str:
    """SHA-256 of a file's content.

    With `memo_dir`, the digest is remembered in `<memo_dir>/.file_hashes.json` under the
    file's absolute path, size and mtime, so a worker does not re-read a 578 MB trace at
    start-up; any change of size or mtime forces a re-hash.
    """
    path = Path(path).resolve()
    st = path.stat()
    tag = f"{st.st_size}:{st.st_mtime_ns}"
    memo_file = Path(memo_dir) / _HASH_MEMO if memo_dir is not None else None
    memo: dict[str, list[str]] = {}
    if memo_file is not None and memo_file.exists():
        try:
            memo = json.loads(memo_file.read_text())
        except (OSError, ValueError):
            memo = {}
        if memo.get(str(path), [None])[0] == tag:
            return memo[str(path)][1]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    digest = h.hexdigest()
    if memo_file is not None:
        memo[str(path)] = [tag, digest]
        memo_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = memo_file.with_name(f"{memo_file.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(memo, indent=1))
        os.replace(tmp, memo_file)
    return digest


def is_complete(d: str | Path) -> bool:
    return (Path(d) / MANIFEST).exists()


def write_once(final: str | Path, write: Callable[[Path], None]) -> Path:
    """Build a cache directory in a temp dir, then rename it into place.

    `write(tmp)` must write `manifest.json` last. Safe with parallel writers: the loser of
    the rename discards its copy.
    """
    final = Path(final)
    if is_complete(final):
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=final.parent, prefix=".tmp-"))
    try:
        write(tmp)
        if not is_complete(tmp):
            raise RuntimeError(f"{write} did not write {MANIFEST}")
        if final.exists() and not is_complete(final):   # a crashed, incomplete copy
            shutil.rmtree(final, ignore_errors=True)
        os.rename(tmp, final)
    except OSError:
        if not is_complete(final):
            raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return final
