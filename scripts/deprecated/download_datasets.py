"""
Download and prepare RNA secondary structure datasets.

Downloads five classic RNA structure datasets to dataset/raw/<name>/.
Handles auto-extraction, failure recovery, and manifest generation.

Datasets:
  ArchiveII   - RNA secondary structure benchmark (auto-extract tar.gz)
  bpRNA       - Large-scale RNA structure database (manual download via README)
  RNAStralign - RNA structural alignment dataset (auto-extract tar.gz if available)
  RNAStrAND   - RNA STRAND database (manual download via README)
  Rfam        - RNA families database (gz files, keep compressed)

Usage:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --datasets ArchiveII Rfam
  python scripts/download_datasets.py --force
  python scripts/download_datasets.py --raw-root D:\\RNA-OmniDiffusion\\dataset\\raw

Dependencies: Python 3.8+ standard library only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import ssl
import sys
import tarfile
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: Dict[str, Dict[str, Any]] = {
    "ArchiveII": {
        "name": "ArchiveII",
        "url": "https://rna.urmc.rochester.edu/pub/archiveII.tar.gz",
        "out_filename": "archiveII.tar.gz",
        "extract": True,
        "extract_format": "tar.gz",
        "description": "ArchiveII RNA secondary structure benchmark",
    },
    "bpRNA": {
        "name": "bpRNA",
        "url": None,
        "out_filename": None,
        "extract": False,
        "manual_url": "http://bprna.cgrb.oregonstate.edu/download.php",
        "manual_note": (
            "bpRNA requires manual download.\n"
            "\n"
            "1. Visit: http://bprna.cgrb.oregonstate.edu/download.php\n"
            "2. Recommended downloads:\n"
            "   - bpRNA-1m (all 1 million+ structures, dot-bracket format)\n"
            "   - bpRNA benchmark sets (Rfam-family-based splits)\n"
            "3. Download the dot-bracket or CT format files.\n"
            "4. Place the downloaded file(s) directly in this directory.\n"
            "\n"
            "After placing files, re-run this script to generate manifest.json.\n"
        ),
        "description": "Large-scale RNA structure database",
    },
    "RNAStralign": {
        "name": "RNAStralign",
        "url": "https://rna.urmc.rochester.edu/pub/RNAStralign.tar.gz",
        "fallback_url": "https://rna.urmc.rochester.edu/pub/",
        "out_filename": "RNAStralign.tar.gz",
        "extract": True,
        "extract_format": "tar.gz",
        "manual_note": (
            "Automatic download of RNAStralign.tar.gz failed.\n"
            "\n"
            "1. Visit: https://rna.urmc.rochester.edu/pub/\n"
            "2. Look for RNAStralign.tar.gz or similar RNAStralign archive.\n"
            "3. Download and place the archive in this directory.\n"
            "4. Re-run this script or manually extract with:\n"
            "   tar -xzf RNAStralign.tar.gz\n"
            "\n"
            "After placing files, re-run this script to update manifest.json.\n"
        ),
        "description": "RNA structural alignment dataset",
    },
    "RNAStrAND": {
        "name": "RNAStrAND",
        "url": None,
        "out_filename": None,
        "extract": False,
        "manual_url": "https://www.rnasoft.ca/strand/downloads.php",
        "manual_note": (
            "RNA STRAND requires manual download.\n"
            "\n"
            "1. Visit: https://www.rnasoft.ca/strand/downloads.php\n"
            "2. Download the RNA STRAND database (CT format files).\n"
            "3. Place the downloaded file(s) directly in this directory.\n"
            "\n"
            "After placing files, re-run this script to generate manifest.json.\n"
        ),
        "description": "RNA STRAND database",
    },
    "Rfam": {
        "name": "Rfam",
        "urls": [
            "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz",
            "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.full_region.gz",
            "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz",
            "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.tar.gz",
        ],
        "out_filename": None,
        "extract": False,
        "description": "RNA families database (seed alignment + full + covariance models)",
    },
    "RNASSTR": {
        "name": "RNASSTR",
        "urls": [
            "https://zenodo.org/records/15319168/files/RNASSTR_train.csv",
            "https://zenodo.org/records/15319168/files/RNASSTR_validation.csv",
            "https://zenodo.org/records/15319168/files/RNASSTR_test.csv",
        ],
        "out_filename": None,
        "extract": False,
        "description": "RNA Secondary Structure Repository — 5M sequences, 4170 Rfam families, CSV format (2025)",
    },
    "CRW_tRNA": {
        "name": "CRW_tRNA",
        "urls": [
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnA.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnC.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnD.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnE.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnF.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnG.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnH.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnI.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnK.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnM.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnN.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnP.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnQ.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnW.ct.tar.gz",
            "http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnY1.ct.tar.gz",
        ],
        "out_filename": None,
        "extract": True,
        "extract_format": "tar.gz",
        "description": "CRW Comparative tRNA dataset (Gutell Lab) — 15 isoacceptors, ~32K sequences, CT format",
    },
    "RNA3DB": {
        "name": "RNA3DB",
        "urls": [
            "https://github.com/marcellszi/rna3db/releases/download/2026-01-05-full-release/rna3db-jsons.tar.gz",
            "https://github.com/marcellszi/rna3db/releases/download/2026-01-05-full-release/rna3db-cmscans.tar.gz",
        ],
        "out_filename": None,
        "extract": True,
        "extract_format": "tar.gz",
        "description": "RNA3DB — PDB-derived non-redundant RNA structures, 216 Rfam families, family-disjoint splits (2024)",
    },
    "CHANRG": {
        "name": "CHANRG",
        "urls": [
            "https://huggingface.co/datasets/multimolecule/chanrg/resolve/main/train.parquet",
            "https://huggingface.co/datasets/multimolecule/chanrg/resolve/main/validation.parquet",
            "https://huggingface.co/datasets/multimolecule/chanrg/resolve/main/test.parquet",
        ],
        "out_filename": None,
        "extract": False,
        "description": "CHANRG — OOD generalization benchmark, Rfam 15.0, 170K sequences, Parquet format (2026)",
    },
}

# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

HTTP_CLIENT_NAME = "RNA-OmniPrefold-downloader/1.0"


def _make_ssl_context() -> ssl.SSLContext:
    """Create a permissive SSL context for servers with certificate issues."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _download_file(url: str, dest: Path, timeout: int = 60) -> Tuple[bool, str]:
    """Download a single file from *url* to *dest*. Returns (ok, error_message)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": HTTP_CLIENT_NAME})
        with urllib.request.urlopen(req, timeout=timeout, context=_make_ssl_context()) as resp:
            with open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=128 * 1024)
        tmp.replace(dest)
        return True, ""
    except urllib.error.HTTPError as e:
        # Clean up partial
        if tmp.exists():
            tmp.unlink()
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return False, str(e)


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(dataset_dir: Path, info: Dict[str, Any]) -> None:
    """Write or update manifest.json in *dataset_dir*."""
    manifest_path = dataset_dir / "manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    manifest = {
        "dataset": info.get("dataset", dataset_dir.name),
        "description": info.get("description", ""),
        "downloads": info.get("downloads", []),
        "extracted": info.get("extracted", False),
        "files": [],
        "errors": info.get("errors", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Preserve any previous download entries merged by source URL
    prev_downloads = existing.get("downloads", []) if isinstance(existing, dict) else []
    known_sources = {d.get("source") for d in manifest["downloads"] if d.get("source")}
    for pd in prev_downloads:
        if pd.get("source") not in known_sources and pd.get("source"):
            manifest["downloads"].append(pd)

    # Also list all current files in the directory
    file_list = []
    if dataset_dir.exists():
        for p in sorted(dataset_dir.rglob("*")):
            if p.is_file() and p.name not in ("manifest.json", "README_DOWNLOAD.txt"):
                rel = p.relative_to(dataset_dir)
                file_list.append({
                    "path": str(rel),
                    "size": p.stat().st_size,
                    "sha256": _compute_sha256(p) if p.stat().st_size < 500 * 1024 * 1024 else "skipped_large",
                })
    manifest["files"] = file_list

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_readme(dataset_dir: Path, note: str) -> None:
    """Write README_DOWNLOAD.txt in *dataset_dir*."""
    readme_path = dataset_dir / "README_DOWNLOAD.txt"
    readme_path.write_text(note.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def _extract_tar_gz(archive_path: Path, dest_dir: Path) -> Tuple[bool, str]:
    """Extract .tar.gz to *dest_dir*. Returns (ok, error_message)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
        return True, ""
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Single dataset handler
# ---------------------------------------------------------------------------

def _process_dataset(name: str, info: Dict[str, Any], raw_root: Path, force: bool) -> Dict[str, Any]:
    """Download/extract a single dataset. Returns a result dict for the summary."""
    dataset_dir = raw_root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "dataset": name,
        "directory": str(dataset_dir),
        "status": "pending",
        "downloads": [],
        "extracted": False,
        "errors": [],
    }

    # --- Handle manual-only datasets first ---
    if info.get("url") is None and info.get("urls") is None:
        note = info.get("manual_note", f"Manual download required for {name}.")
        _write_readme(dataset_dir, note)
        # Build manifest with manual status
        manifest_info = {
            "dataset": name,
            "description": info.get("description", ""),
            "downloads": [{"source": info.get("manual_url", "N/A"), "status": "manual_required"}],
            "extracted": False,
            "errors": [],
        }
        # Check if files already exist from previous manual download
        existing_files = list(dataset_dir.rglob("*"))
        has_content = any(
            f.is_file() and f.name not in ("manifest.json", "README_DOWNLOAD.txt")
            for f in existing_files
        )
        if has_content:
            result["status"] = "manual_complete"
            result["errors"].append("manual_download_already_present")
        else:
            result["status"] = "manual_required"
        _write_manifest(dataset_dir, manifest_info)
        return result

    # --- Handle single-URL datasets (ArchiveII, RNAStralign) ---
    if "url" in info and info["url"]:
        url = info["url"]
        out_filename = info.get("out_filename") or Path(url).name
        dest = dataset_dir / out_filename

        if dest.exists() and not force:
            result["status"] = "cached"
            result["downloads"].append({
                "source": url,
                "local": str(dest),
                "status": "cached",
                "size": dest.stat().st_size,
            })
        else:
            ok, err = _download_file(url, dest)
            result["downloads"].append({
                "source": url,
                "local": str(dest),
                "status": "success" if ok else "failed",
                "size": dest.stat().st_size if ok else 0,
            })
            if not ok:
                result["errors"].append(f"Download failed: {err}")
                # Try fallback for RNAStralign
                if name == "RNAStralign" and info.get("fallback_url"):
                    note = info.get("manual_note", f"Manual download required for {name}.")
                    _write_readme(dataset_dir, note)
                    result["status"] = "download_failed_manual_required"
            else:
                result["status"] = "downloaded"

        # Extract if applicable
        if info.get("extract") and dest.exists():
            if name == "ArchiveII":
                # ArchiveII extracts into its own directory; check for marker file
                already_extracted = (dataset_dir / "ArchiveII").exists() or any(
                    f.suffix.lower() in (".ct", ".bpseq", ".fasta", ".txt")
                    for f in dataset_dir.iterdir()
                    if f.is_file() and f.name != out_filename
                )
            else:
                # Generic check: if there are files besides the archive
                already_extracted = any(
                    f.name != out_filename and f.name not in ("manifest.json", "README_DOWNLOAD.txt")
                    for f in dataset_dir.rglob("*") if f.is_file()
                )

            if already_extracted and not force:
                result["extracted"] = True
            else:
                ok, err = _extract_tar_gz(dest, dataset_dir)
                if ok:
                    result["extracted"] = True
                else:
                    result["errors"].append(f"Extract failed: {err}")

            if result["extracted"] and result["status"] in ("cached", "downloaded"):
                result["status"] = "ready"

    # --- Handle multi-URL datasets (Rfam) ---
    elif "urls" in info:
        all_ok = True
        for url in info["urls"]:
            filename = Path(url).name
            dest = dataset_dir / filename

            if dest.exists() and not force:
                result["downloads"].append({
                    "source": url,
                    "local": str(dest),
                    "status": "cached",
                    "size": dest.stat().st_size,
                })
            else:
                ok, err = _download_file(url, dest)
                dl_status = "success" if ok else "failed"
                result["downloads"].append({
                    "source": url,
                    "local": str(dest),
                    "status": dl_status,
                    "size": dest.stat().st_size if ok else 0,
                })
                if not ok:
                    result["errors"].append(f"Download failed ({filename}): {err}")
                    all_ok = False

        if all_ok:
            result["status"] = "ready"
        elif any(d["status"] == "success" for d in result["downloads"]):
            result["status"] = "partial"
        else:
            result["status"] = "download_failed"

        # Extract tar.gz files if dataset requires extraction
        if info.get("extract"):
            for dl in result["downloads"]:
                if dl["status"] not in ("success", "cached"):
                    continue
                src = Path(dl["local"])
                if not src.exists() or not src.suffix.lower() in (".gz",):
                    continue
                if src.name.endswith(".tar.gz"):
                    # Check if already extracted
                    stem = src.name.replace(".tar.gz", "")
                    already = (dataset_dir / stem).exists() or any(
                        f.name not in ("manifest.json", "README_DOWNLOAD.txt")
                        and f.name != src.name
                        for f in dataset_dir.rglob("*") if f.is_file()
                    )
                    if already and not force:
                        result["extracted"] = True
                        continue
                    ok, err = _extract_tar_gz(src, dataset_dir)
                    if ok:
                        result["extracted"] = True
                    else:
                        result["errors"].append(f"Extract failed ({src.name}): {err}")
            if result["extracted"] and result["status"] in ("ready", "partial"):
                result["status"] = "ready"

    # --- Write manifest ---
    manifest_info = {
        "dataset": name,
        "description": info.get("description", ""),
        "downloads": result["downloads"],
        "extracted": result["extracted"],
        "errors": result["errors"],
    }
    _write_manifest(dataset_dir, manifest_info)

    return result


# ---------------------------------------------------------------------------
# Summary & reporting
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary table of all dataset processing results."""
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    col_name = 16
    col_status = 22
    col_errors = 28

    header = f"{'Dataset':<{col_name}} {'Status':<{col_status}} {'Notes'}"
    print(header)
    print("-" * 70)

    ok_count = 0
    fail_count = 0
    manual_count = 0

    for r in results:
        name = r["dataset"]
        status = r["status"]
        errors = r.get("errors", [])

        if status in ("ready", "downloaded", "cached"):
            ok_count += 1
            status_display = status
        elif status == "manual_required":
            manual_count += 1
            status_display = "MANUAL DOWNLOAD"
        elif status == "manual_complete":
            ok_count += 1
            status_display = "manual (files present)"
        elif status == "partial":
            ok_count += 1
            status_display = "partial"
        else:
            fail_count += 1
            status_display = status

        notes = ""
        if errors:
            notes = "; ".join(errors[:2])
            if len(errors) > 2:
                notes += f" (+{len(errors) - 2} more)"

        print(f"{name:<{col_name}} {status_display:<{col_status}} {notes}")

    print("-" * 70)
    total = len(results)
    print(f"Total: {total}  |  OK: {ok_count}  |  Manual: {manual_count}  |  Failed: {fail_count}")
    print("=" * 70)

    if manual_count > 0:
        print("\nMANUAL DOWNLOADS REQUIRED:")
        for r in results:
            if r["status"] in ("manual_required", "download_failed_manual_required"):
                print(f"  - {r['dataset']}: see {r['directory']}/README_DOWNLOAD.txt")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download RNA secondary structure datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_datasets.py\n"
            "  python scripts/download_datasets.py --datasets ArchiveII Rfam\n"
            "  python scripts/download_datasets.py --force\n"
            "  python scripts/download_datasets.py --raw-root D:\\RNA-OmniDiffusion\\dataset\\raw\n"
        ),
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Root directory for raw datasets (default: <repo>/dataset/raw)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        choices=list(DATASETS.keys()) + [k.lower() for k in DATASETS],
        help="Specific datasets to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Download timeout in seconds (default: 120)",
    )

    args = parser.parse_args()

    # Determine raw root
    if args.raw_root:
        raw_root = Path(args.raw_root).resolve()
    else:
        # Default: <repo_root>/dataset/raw
        repo_root = Path(__file__).resolve().parents[1]
        raw_root = repo_root / "dataset" / "raw"

    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"Raw dataset root: {raw_root}")

    # Select datasets
    if args.datasets:
        selected = []
        for ds in args.datasets:
            # Case-insensitive matching
            match = None
            for k in DATASETS:
                if k.lower() == ds.lower():
                    match = k
                    break
            if match:
                selected.append(match)
            else:
                print(f"Warning: unknown dataset '{ds}', skipping")
        if not selected:
            print("No valid datasets selected. Available: " + ", ".join(DATASETS.keys()))
            sys.exit(1)
    else:
        selected = list(DATASETS.keys())

    print(f"Selected datasets: {', '.join(selected)}")
    if args.force:
        print("Force mode: will re-download all files.")

    # Process each dataset
    results = []
    for name in selected:
        info = DATASETS[name]
        print(f"\n--- Processing {name} ({info['description']}) ---")
        try:
            result = _process_dataset(name, info, raw_root, args.force)
            results.append(result)
            print(f"  Status: {result['status']}")
            for dl in result.get("downloads", []):
                status_icon = "OK" if dl["status"] in ("success", "cached") else "FAIL"
                print(f"  [{status_icon}] {dl.get('source', 'N/A')}")
                if dl["status"] == "success":
                    print(f"       -> {dl['local']} ({dl['size']:,} bytes)")
            if result.get("extracted"):
                print(f"  Extracted: yes")
            for err in result.get("errors", []):
                print(f"  Error: {err}")
        except Exception as e:
            print(f"  FATAL: {e}")
            traceback.print_exc()
            results.append({
                "dataset": name,
                "directory": str(raw_root / name),
                "status": "fatal_error",
                "downloads": [],
                "extracted": False,
                "errors": [str(e)],
            })

    _print_summary(results)


if __name__ == "__main__":
    main()
