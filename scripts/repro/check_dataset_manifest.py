#!/usr/bin/env python3
"""Check or generate dataset sha256 manifest.
校验或生成数据集哈希清单。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build(dataset_root: Path) -> dict[str, str]:
    files = {}
    for p in sorted(dataset_root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(dataset_root).as_posix()
            files[rel] = sha256_file(p)
    return files


def parse_manifest(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "  " not in line:
            raise ValueError(f"Invalid manifest line: {line}")
        h, rel = line.split("  ", 1)
        out[rel] = h
    return out


def write_manifest(path: Path, pairs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{pairs[k]}  {k}\n" for k in sorted(pairs)), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check/generate dataset manifest")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--create-if-missing", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    manifest = Path(args.manifest)

    if not dataset_root.exists():
        print(f"[FAIL] Dataset root not found: {dataset_root}")
        return 1

    actual = build(dataset_root)
    if not manifest.exists() or not parse_manifest(manifest):
        if args.create_if_missing:
            write_manifest(manifest, actual)
            print(f"[INFO] Manifest created: {manifest} ({len(actual)} files)")
            return 0
        print(f"[FAIL] Manifest missing or empty: {manifest}")
        return 1

    expected = parse_manifest(manifest)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatch = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])

    if missing or extra or mismatch:
        print(f"[FAIL] Dataset manifest mismatch: {manifest}")
        if missing:
            print(f"  missing({len(missing)}): {missing[:10]}")
        if extra:
            print(f"  extra({len(extra)}): {extra[:10]}")
        if mismatch:
            print(f"  hash_mismatch({len(mismatch)}): {mismatch[:10]}")
        return 1

    print(f"[PASS] Dataset manifest verified: {manifest} ({len(actual)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
