#!/usr/bin/env python3
"""Verify vendored DeepLens lock integrity.
校验 vendored DeepLens 快照完整性与哈希一致性。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "  " not in line:
            raise ValueError(f"Invalid manifest line: {line}")
        digest, rel = line.split("  ", 1)
        pairs[rel] = digest
    return pairs


def build_actual(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.endswith(".pyc") or "__pycache__" in rel:
            continue
        out[rel] = sha256_file(p)
    return out


def tree_hash(pairs: dict[str, str]) -> str:
    payload = "".join(f"{pairs[k]}  {k}\n" for k in sorted(pairs)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_import(root: Path) -> None:
    try:
        import deeplens  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Failed to import deeplens: {exc}") from exc

    package_file = Path(getattr(deeplens, "__file__", "")).resolve()
    expected_pkg_root = (root / "deeplens").resolve()
    if expected_pkg_root not in package_file.parents:
        raise RuntimeError(
            "deeplens import path mismatch: "
            f"got {package_file}, expected under {expected_pkg_root}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify vendored DeepLens lock")
    parser.add_argument(
        "--root", default="third_party/deeplens-core", help="Vendored DeepLens root"
    )
    parser.add_argument(
        "--lock", default="repro/deeplens_lock.json", help="Lock metadata JSON"
    )
    parser.add_argument(
        "--manifest",
        default="repro/deeplens_sha256_manifest.txt",
        help="SHA256 manifest path",
    )
    parser.add_argument(
        "--strict-import",
        action="store_true",
        help="Also verify imported deeplens comes from vendored path",
    )
    args = parser.parse_args()

    root = Path(args.root)
    lock_path = Path(args.lock)
    manifest_path = Path(args.manifest)

    if not root.exists():
        print(f"[FAIL] Missing root: {root}")
        return 1
    if not lock_path.exists():
        print(f"[FAIL] Missing lock json: {lock_path}")
        return 1
    if not manifest_path.exists():
        print(f"[FAIL] Missing manifest: {manifest_path}")
        return 1

    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        expected = read_manifest(manifest_path)
        actual = build_actual(root)
    except Exception as exc:
        print(f"[FAIL] Parse/build error: {exc}")
        return 1

    ok = True

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatch = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])

    if missing:
        ok = False
        print(f"[FAIL] Missing files ({len(missing)}): {missing[:10]}")
    if extra:
        ok = False
        print(f"[FAIL] Extra files ({len(extra)}): {extra[:10]}")
    if mismatch:
        ok = False
        print(f"[FAIL] Hash mismatch ({len(mismatch)}): {mismatch[:10]}")

    actual_tree = tree_hash(actual)
    lock_tree = str(lock.get("tree_sha256", ""))
    if lock_tree and actual_tree != lock_tree:
        ok = False
        print(f"[FAIL] Tree hash mismatch: actual={actual_tree}, lock={lock_tree}")

    if str(lock.get("version", "")) != "1.2.3":
        print("[WARN] Lock version is not 1.2.3; verify snapshot source carefully.")

    if args.strict_import:
        try:
            verify_import(root)
        except Exception as exc:
            ok = False
            print(f"[FAIL] Import verification failed: {exc}")

    if not ok:
        return 1

    print("[PASS] DeepLens lock verification passed")
    print(f"[INFO] file_count={len(actual)} tree_sha256={actual_tree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
