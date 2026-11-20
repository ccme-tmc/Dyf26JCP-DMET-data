#!/usr/bin/env python3
"""Rebuild the ``SMMRT-DMET-ST-data`` archive from the author's origin folder.

The origin folder holds the raw working trees of the three compounds plus the
spatial-truncation study.  It also holds run-time scraps (scheduler stdout,
bytecode caches, binary checkpoints) and ~100 MB of legacy Molcas dumps that we
do not publish.  This script applies the documented rules in ``rules.yaml``,
copies what survives into this repository, and writes ``manifest.csv`` so that
every archived byte can be verified later.

Usage
-----
    python tools/build_archive.py --source <Origindata> --dest .
    python tools/build_archive.py --source <Origindata> --dest . --dry-run
    python tools/build_archive.py --check

``--dry-run`` prints what would be copied and nothing is written.
``--check`` re-hashes the files listed in ``manifest.csv`` (against both this
repository and, if given, ``--source``) and exits non-zero on any mismatch.

Only the standard library plus PyYAML is required (PyYAML is already a
dependency of the companion code repository).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

try:
    import yaml
except ImportError:  # pragma: no cover - helpful message instead of a traceback
    sys.exit("build_archive.py needs PyYAML: python -m pip install pyyaml")

MANIFEST = "manifest.csv"
MAX_DEFAULT = 50 * 1024 * 1024
SYSTEM_DIRS = ("Dyacac3_cell_Analyse", "DybbBr", "DyCp_Orcapyscf", "Spacetruncation")
REPO_META = {
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "EXCLUDED.md",
    "LICENSE",
    "README.md",
    MANIFEST,
}
REPO_META_DIRS = {"tools"}
CHUNK = 1 << 20


# --------------------------------------------------------------------------- #
# glob matching (gitignore-flavoured, enough for the rules below)
# --------------------------------------------------------------------------- #
_GLOB_CACHE: dict[str, re.Pattern[str]] = {}


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    i, out = 0, []
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append(r"(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(r".*")
            i += 2
        elif pattern[i] == "*":
            out.append(r"[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append(r"[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _matches(pattern: str, rel_posix: str, name: str) -> bool:
    """A pattern with a slash matches the relative path, otherwise the name."""
    rx = _GLOB_CACHE.get(pattern)
    if rx is None:
        rx = _GLOB_CACHE[pattern] = _glob_to_regex(pattern)
    return bool(rx.match(rel_posix if "/" in pattern else name))


# --------------------------------------------------------------------------- #
# categorisation
# --------------------------------------------------------------------------- #
def classify(rel: PurePosixPath) -> str:
    """Return a coarse kind used in manifest.csv and the README tables."""
    name = rel.name
    suffix = rel.suffix.lower()
    parents = {p.lower() for p in rel.parts[:-1]}
    if suffix in {".py", ".sh", ".ipynb"}:
        return "script"
    if suffix in {".yml", ".yaml"}:
        return "config"
    if name.upper().startswith("CFPS") and suffix == ".out":
        return "cfp"
    if name.upper().startswith("VP") and suffix == ".out":
        return "spc"
    if suffix in {".tp", ".tps"}:
        return "tp_matrix"
    if "tps_file" in parents or "tps_v1" in parents or "tpsv1" in parents:
        return "tau"
    if name.startswith("freq") and suffix == ".log":
        return "phonon"
    if suffix == ".hess":
        return "phonon"
    if name in {"POSCAR", "Atom_list", "dpxs"} or suffix in {".xyz", ".vasp"}:
        return "structure"
    if suffix == ".eso" or name == "gtensor.out":
        return "tensor"
    if suffix in {".png", ".eps", ".jpg", ".pdf"}:
        return "figure"
    if suffix == ".log":
        return "log"
    return "other"


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #
def load_rules(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        sys.exit(f"{path}: expected a mapping")
    rules = {"max_bytes": int(data.get("max_bytes") or MAX_DEFAULT), "exclude": []}
    for entry in data.get("exclude") or []:
        if "pattern" not in entry:
            sys.exit(f"{path}: every exclude entry needs a 'pattern'")
        rules["exclude"].append(
            {"pattern": str(entry["pattern"]), "reason": str(entry.get("reason", ""))}
        )
    return rules


def exclusion_reason(rel: PurePosixPath, size: int, rules: dict) -> str | None:
    rel_posix = rel.as_posix()
    if size > rules["max_bytes"]:
        return f"larger than {rules['max_bytes'] // (1024 * 1024)} MB safety limit"
    for rule in rules["exclude"]:
        if _matches(rule["pattern"], rel_posix, rel.name):
            return rule["reason"] or rule["pattern"]
    return None


# --------------------------------------------------------------------------- #
# hashing / copying
# --------------------------------------------------------------------------- #
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def scan(source: Path, rules: dict):
    """Split the origin tree into (kept, dropped[{reason, bytes}])."""
    kept, dropped = [], {}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = PurePosixPath(path.relative_to(source).as_posix())
        size = path.stat().st_size
        reason = exclusion_reason(rel, size, rules)
        if reason:
            slot = dropped.setdefault(reason, {"files": 0, "bytes": 0})
            slot["files"] += 1
            slot["bytes"] += size
        else:
            kept.append((rel, path, size))
    return kept, dropped


def write_manifest(dest: Path, rows: list[dict]) -> None:
    with (dest / MANIFEST).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "system", "kind", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def summarise(kept, dropped, rules) -> None:
    total = sum(size for _, _, size in kept)
    print(f"files to archive : {len(kept)}")
    print(f"bytes            : {total} ({total / 1048576:.1f} MiB)")
    systems, kinds = {}, {}
    for rel, _, size in kept:
        systems.setdefault(rel.parts[0], [0, 0])
        systems[rel.parts[0]][0] += 1
        systems[rel.parts[0]][1] += size
        kind = classify(rel)
        kinds[kind] = kinds.get(kind, 0) + 1
    print("per system       :")
    for name, (count, size) in sorted(systems.items()):
        print(f"  {name:26} {count:5} files  {size / 1048576:8.1f} MiB")
    print("per kind         : " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    sizes = sorted(((size, rel) for rel, _, size in kept), reverse=True)[:5]
    print("largest files    :")
    for size, rel in sizes:
        print(f"  {size / 1048576:7.2f} MiB  {rel}")
    if dropped:
        print("excluded         :")
        for reason, slot in sorted(dropped.items(), key=lambda kv: -kv[1]["bytes"]):
            print(f"  {slot['files']:4} files  {slot['bytes'] / 1048576:8.1f} MiB  {reason}")
    print(f"largest allowed  : {rules['max_bytes'] / 1048576:.0f} MiB")


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def cmd_build(source: Path, dest: Path, rules: dict, force: bool) -> int:
    kept, dropped = scan(source, rules)
    summarise(kept, dropped, rules)
    clashes = [name for name in SYSTEM_DIRS if (dest / name).exists()]
    if clashes and not force:
        print(
            f"error: {dest} already contains {', '.join(clashes)}; "
            "pass --force to overwrite (files are replaced, nothing is deleted)",
            file=sys.stderr,
        )
        return 2
    rows = []
    for index, (rel, src, size) in enumerate(kept, 1):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        rows.append(
            {
                "path": rel.as_posix(),
                "system": rel.parts[0],
                "kind": classify(rel),
                "bytes": size,
                "sha256": sha256_of(target),
            }
        )
        if index % 250 == 0:
            print(f"  copied {index}/{len(kept)}", flush=True)
    write_manifest(dest, rows)
    print(f"wrote {MANIFEST} with {len(rows)} rows")
    return 0


def cmd_check(dest: Path, source: Path | None) -> int:
    manifest = dest / MANIFEST
    if not manifest.is_file():
        print(f"error: {manifest} not found", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    problems, checked, total = [], 0, 0
    for row in rows:
        total += int(row["bytes"])
        local = dest / row["path"]
        if not local.is_file():
            problems.append(f"missing in repository: {row['path']}")
            continue
        if local.stat().st_size != int(row["bytes"]):
            problems.append(f"size mismatch: {row['path']}")
            continue
        if sha256_of(local) != row["sha256"]:
            problems.append(f"sha256 mismatch: {row['path']}")
            continue
        if source is not None:
            original = source / row["path"]
            if not original.is_file():
                problems.append(f"missing in source: {row['path']}")
                continue
            if sha256_of(original) != row["sha256"]:
                problems.append(f"source sha256 mismatch: {row['path']}")
                continue
        checked += 1
    listed = {row["path"] for row in rows}
    for path in sorted(dest.rglob("*")):
        if not path.is_file():
            continue
        rel = PurePosixPath(path.relative_to(dest).as_posix())
        if rel.as_posix() in listed:
            continue
        if rel.parts[0] in REPO_META_DIRS or (len(rel.parts) == 1 and rel.name in REPO_META):
            continue
        if rel.as_posix().startswith(".git/"):
            continue
        problems.append(f"unlisted file in repository: {rel.as_posix()}")
    print(f"manifest rows    : {len(rows)}")
    print(f"bytes in manifest: {total} ({total / 1048576:.1f} MiB)")
    print(f"verified         : {checked}")
    print(f"problems         : {len(problems)}")
    for problem in problems[:40]:
        print("  " + problem)
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more")
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="stage / verify the SMMRT-DMET-ST-data archive")
    parser.add_argument("--source", type=Path, help="origin data folder (e.g. .../Origindata)")
    parser.add_argument("--dest", type=Path, default=here, help="repository root (default: this repo)")
    parser.add_argument("--rules", type=Path, default=here / "tools" / "rules.yaml")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument("--check", action="store_true", help="verify manifest.csv, write nothing")
    parser.add_argument("--force", action="store_true", help="allow overwriting an already staged tree")
    args = parser.parse_args(argv)

    dest = args.dest.resolve()
    if args.check:
        return cmd_check(dest, args.source.resolve() if args.source else None)
    if not args.source:
        parser.error("--source is required unless --check is used")
    source = args.source.resolve()
    if not source.is_dir():
        parser.error(f"--source is not a directory: {source}")
    if not args.rules.is_file():
        parser.error(f"--rules not found: {args.rules}")
    rules = load_rules(args.rules)
    if args.dry_run:
        kept, dropped = scan(source, rules)
        summarise(kept, dropped, rules)
        return 0
    return cmd_build(source, dest, rules, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
