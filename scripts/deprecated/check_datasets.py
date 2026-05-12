"""Local dataset metadata checker for RNA-OmniPrefold.

Checks dataset existence, file counts, sizes, and manifest status
without reading large file contents. Also supports comparing two
dataset check reports (local vs remote) without connecting remotely.

Usage:
    python scripts/check_datasets.py --root dataset/raw --out outputs/dataset_check/local
    python scripts/check_datasets.py compare --local-report outputs/dataset_check/local/dataset_check.json --remote-report outputs/dataset_check/remote/dataset_check.json --out outputs/dataset_check/compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def check_dataset_dir(root: Path) -> dict[str, Any]:
    """Scan a dataset root directory and return metadata."""
    if not root.exists():
        return {"root": str(root), "exists": False, "datasets": []}

    datasets: list[dict[str, Any]] = []
    total_files = 0
    total_size = 0

    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith(".") or entry.name == ".gitkeep":
            continue

        info: dict[str, Any] = {
            "name": entry.name,
            "exists": True,
            "file_count": 0,
            "total_size_bytes": 0,
            "extensions": {},
            "has_manifest": (entry / "manifest.json").exists(),
            "has_readme": (entry / "README_DOWNLOAD.txt").exists(),
            "sample_files": [],
        }

        for f in sorted(entry.rglob("*")):
            if f.is_file() and f.name not in ("manifest.json", "README_DOWNLOAD.txt", ".gitkeep"):
                size = f.stat().st_size
                info["file_count"] += 1
                info["total_size_bytes"] += size
                ext = f.suffix.lower() or "(none)"
                info["extensions"][ext] = info["extensions"].get(ext, 0) + 1
                if len(info["sample_files"]) < 10:
                    info["sample_files"].append({
                        "path": str(f.relative_to(root / entry.name)),
                        "size": size,
                    })

        info["total_size_mb"] = round(info["total_size_bytes"] / (1024 * 1024), 2)
        datasets.append(info)
        total_files += info["file_count"]
        total_size += info["total_size_bytes"]

    return {
        "root": str(root),
        "exists": True,
        "dataset_count": len(datasets),
        "total_files": total_files,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "datasets": datasets,
    }


def compare_reports(local: dict, remote: dict) -> dict[str, Any]:
    """Compare two dataset check reports."""
    local_ds = {d["name"]: d for d in local.get("datasets", [])}
    remote_ds = {d["name"]: d for d in remote.get("datasets", [])}
    all_names = sorted(set(local_ds.keys()) | set(remote_ds.keys()))

    diffs: list[dict[str, Any]] = []
    for name in all_names:
        l = local_ds.get(name)
        r = remote_ds.get(name)
        diff: dict[str, Any] = {"name": name}

        if l is None:
            diff["status"] = "missing_local"
            diffs.append(diff)
            continue
        if r is None:
            diff["status"] = "missing_remote"
            diffs.append(diff)
            continue

        file_diff = l.get("file_count", 0) - r.get("file_count", 0)
        size_diff = l.get("total_size_bytes", 0) - r.get("total_size_bytes", 0)
        manifest_diff = (l.get("has_manifest", False) != r.get("has_manifest", False))

        if file_diff == 0 and abs(size_diff) < 1024 and not manifest_diff:
            diff["status"] = "match"
        else:
            diff["status"] = "mismatch"
            diff["file_count_diff"] = file_diff
            diff["size_diff_bytes"] = size_diff
            diff["size_diff_mb"] = round(size_diff / (1024 * 1024), 2)
            diff["manifest_diff"] = manifest_diff
        diffs.append(diff)

    return {
        "local_root": local.get("root"),
        "remote_root": remote.get("root"),
        "differences": diffs,
        "match_count": sum(1 for d in diffs if d["status"] == "match"),
        "mismatch_count": sum(1 for d in diffs if d["status"] == "mismatch"),
        "missing_local": sum(1 for d in diffs if d["status"] == "missing_local"),
        "missing_remote": sum(1 for d in diffs if d["status"] == "missing_remote"),
    }


def write_report(out_dir: Path, data: dict, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md_lines = [f"# Dataset Check: {stem}", ""]
    if "datasets" in data:
        md_lines.append(f"Root: {data.get('root')}")
        md_lines.append(f"Datasets: {data.get('dataset_count', 0)}")
        md_lines.append(f"Total files: {data.get('total_files', 0)}")
        md_lines.append(f"Total size: {data.get('total_size_mb', 0)} MB")
        md_lines.append("")
        for ds in data["datasets"]:
            icon = "[OK]" if ds["exists"] else "[MISSING]"
            md_lines.append(
                f"{icon} {ds['name']}: {ds['file_count']} files, "
                f"{ds['total_size_mb']} MB, manifest={'yes' if ds['has_manifest'] else 'no'}"
            )
    elif "differences" in data:
        md_lines.append(f"Local: {data.get('local_root')}")
        md_lines.append(f"Remote: {data.get('remote_root')}")
        md_lines.append(f"Match: {data['match_count']} | Mismatch: {data['mismatch_count']}")
        md_lines.append(f"Missing local: {data['missing_local']} | Missing remote: {data['missing_remote']}")
        md_lines.append("")
        for d in data["differences"]:
            icon = {"match": "=", "mismatch": "!=", "missing_local": "?L", "missing_remote": "?R"}.get(d["status"], "??")
            detail = ""
            if d["status"] == "mismatch":
                detail = f" (files: {d.get('file_count_diff', 0)}, size: {d.get('size_diff_mb', 0)} MB)"
            md_lines.append(f"{icon} {d['name']}{detail}")

    (out_dir / f"{stem}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RNA-OmniPrefold dataset metadata checker"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    check = sub.add_parser("check", help="Check a dataset root directory")
    check.add_argument("--root", required=True, help="Dataset root directory")
    check.add_argument("--out", default="outputs/dataset_check/local", help="Output directory")
    check.set_defaults(func=_run_check)

    compare = sub.add_parser("compare", help="Compare two dataset check reports")
    compare.add_argument("--local-report", required=True, help="Path to local check report JSON")
    compare.add_argument("--remote-report", required=True, help="Path to remote check report JSON")
    compare.add_argument("--out", default="outputs/dataset_check/compare", help="Output directory")
    compare.set_defaults(func=_run_compare)

    args = parser.parse_args()
    args.func(args)


def _run_check(args: argparse.Namespace) -> None:
    root = Path(args.root)
    out = Path(args.out)
    data = check_dataset_dir(root)
    write_report(out, data, "dataset_check")
    print(f"Dataset check -> {out}")
    print(f"  Root: {root}")
    print(f"  Datasets: {data.get('dataset_count', 0)}")
    print(f"  Total: {data.get('total_files', 0)} files, {data.get('total_size_mb', 0)} MB")


def _run_compare(args: argparse.Namespace) -> None:
    local_path = Path(args.local_report)
    remote_path = Path(args.remote_report)
    if not local_path.exists():
        raise SystemExit(f"Local report not found: {local_path}")
    if not remote_path.exists():
        raise SystemExit(f"Remote report not found: {remote_path}")

    local_data = json.loads(local_path.read_text(encoding="utf-8"))
    remote_data = json.loads(remote_path.read_text(encoding="utf-8"))
    result = compare_reports(local_data, remote_data)
    write_report(Path(args.out), result, "compare")
    print(f"Dataset compare -> {args.out}")
    print(f"  Match: {result['match_count']} | Mismatch: {result['mismatch_count']}")
    print(f"  Missing local: {result['missing_local']} | Missing remote: {result['missing_remote']}")


if __name__ == "__main__":
    main()
