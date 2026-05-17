from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parts(*items: str) -> str:
    return "".join(items)

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt"}
EXCLUDED_DIRS = {".git", "outputs", "dataset", "external", "checkpoints", "__pycache__"}
EXCLUDED_PATH_PREFIXES = ("scripts/remote/",)
BLOCKED_STAGE_PREFIXES = (
    "outputs/",
    "dataset/teacher_emb/",
    "dataset/unlabeled/",
)
BLOCKED_STAGE_EXACT = {
    ".env",
    parts("external/model", ".safetensors"),
    "bpRNA_1m_90.zip",
    "rnacentral_active.fasta.gz",
}
BLOCKED_STAGE_SUFFIXES = (
    ".npy",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
)


LEGACY_ASSISTANT_RESIDUALS = [
    parts("ag", "ent"),
    parts("ag", "ent.py"),
    parts("ag", "ent.cmd"),
    parts("ag", "ent.sh"),
    parts("scripts/", "l", "lm.py"),
    "scripts/semantic.py",
    parts("docs/", "ag", "ent.md"),
    parts("docs/", "ag", "ent_guide.md"),
    parts("docs/", "l", "lm_semantic_plan.md"),
    parts("docs/", "l", "lm_negative_result.md"),
]
MAINLINE_DOCS = [
    "README.md",
    "INDEX.md",
    "docs/usage.md",
]
FORBIDDEN_MAINLINE_PHRASES = [
    parts("Deep", "Seek ", "Ag", "ent"),
    parts("LL", "M ", "Ag", "ent"),
    parts("Ag", "ent shell"),
    parts("LL", "M semantic token"),
    parts("LL", "M-powered predictor"),
    parts("Ag", "ent improves performance"),
    parts("scripts/", "l", "lm.py"),
    parts("python ", "ag", "ent.py"),
    parts("python -m ", "ag", "ent"),
]
REMOVED_MAINLINE_PATHS = [
    "models/teacher",
    parts("scripts/extract_", "rna", "fm_embeddings.py"),
    parts("config/seq_pretrain_", "rna", "fm.yaml"),
    parts("config/candidate_from_", "rna", "fm_pretrain.yaml"),
    parts("config/candidate_from_", "rna", "fm_pretrain_msmprm.yaml"),
]
FORBIDDEN_MAINLINE_PATTERNS = [
    re.compile(
        "|".join(
            [
                parts("RNA", "-", "FM"),
                parts("rna", "fm"),
                parts("distill", "_head"),
                parts("lambda", "_distill"),
                parts("teacher", "_embedding"),
                parts("model", r"\.safetensors"),
            ]
        ),
        re.IGNORECASE,
    ),
]
SENSITIVE_PATTERN = re.compile(
    "|".join([
        parts("pass", "word"),
        parts("api[_-]?", "key"),
        parts("sec", "ret"),
        parts("connect", r"\.nmb1"),
        parts("ssh", r"\s+-p"),
    ]),
    re.IGNORECASE,
)


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def run_git(args: list[str]) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def iter_project_text_files() -> list[Path]:
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = rel(path)
        if any(part in EXCLUDED_DIRS for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        if any(relative.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"README.md", "INDEX.md"}:
            files.append(path)
    return files


def staged_files() -> list[str]:
    return run_git(["diff", "--cached", "--name-only"])


def staged_large_or_blocked() -> list[str]:
    blocked: list[str] = []
    for item in staged_files():
        normalized = item.replace("\\", "/")
        if normalized in BLOCKED_STAGE_EXACT:
            blocked.append(normalized)
            continue
        if normalized.startswith(BLOCKED_STAGE_PREFIXES):
            blocked.append(normalized)
            continue
        if normalized.endswith(BLOCKED_STAGE_SUFFIXES):
            blocked.append(normalized)
            continue
        if normalized.startswith("dataset/processed/"):
            path = PROJECT_ROOT / normalized
            if path.exists() and path.stat().st_size > 1_000_000:
                blocked.append(normalized)
    return sorted(set(blocked))


def current_sensitive_hits() -> list[str]:
    hits: list[str] = []
    for path in iter_project_text_files():
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            if SENSITIVE_PATTERN.search(line):
                hits.append(f"{rel(path)}:{line_no}")
    return hits


def residual_legacy_assistant_files() -> list[str]:
    residuals: list[str] = []
    for item in LEGACY_ASSISTANT_RESIDUALS:
        path = PROJECT_ROOT / item
        if path.exists():
            residuals.append(item)
    return residuals


def forbidden_doc_hits() -> list[str]:
    hits: list[str] = []
    for item in MAINLINE_DOCS:
        path = PROJECT_ROOT / item
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for phrase in FORBIDDEN_MAINLINE_PHRASES:
            if phrase in text:
                hits.append(f"{item}: {phrase}")
    return hits


def removed_mainline_residuals() -> list[str]:
    return [item for item in REMOVED_MAINLINE_PATHS if (PROJECT_ROOT / item).exists()]


def forbidden_mainline_hits() -> list[str]:
    hits: list[str] = []
    for path in iter_project_text_files():
        relative = rel(path)
        if relative == "docs/negative.md":
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            if any(pattern.search(line) for pattern in FORBIDDEN_MAINLINE_PATTERNS):
                hits.append(f"{relative}:{line_no}")
    return hits


def candidate_diff_exists() -> bool:
    return bool(run_git(["diff", "--", "config/candidate.yaml"]))


def run_clean(args: argparse.Namespace) -> None:
    warnings: list[str] = []
    report = {
        "candidate_yaml_has_diff": candidate_diff_exists(),
        "blocked_staged_files": staged_large_or_blocked(),
        "sensitive_hits": current_sensitive_hits(),
        "legacy_assistant_residual_files": residual_legacy_assistant_files(),
        "forbidden_mainline_doc_hits": forbidden_doc_hits(),
        "removed_mainline_residuals": removed_mainline_residuals(),
        "forbidden_removed_stack_hits": forbidden_mainline_hits(),
    }

    if report["candidate_yaml_has_diff"]:
        warnings.append("config/candidate.yaml has local diff")
    if report["blocked_staged_files"]:
        warnings.append("blocked generated, model, dataset, or sensitive files are staged")
    if report["sensitive_hits"]:
        warnings.append("possible sensitive credential or remote-login text found in current project files")
    if report["legacy_assistant_residual_files"]:
        warnings.append("legacy assistant or language-model semantic files remain in the working tree")
    if report["forbidden_mainline_doc_hits"]:
        warnings.append("forbidden legacy assistant or language-model framing remains in mainline docs")
    if report["removed_mainline_residuals"]:
        warnings.append("removed mainline files or directories still exist")
    if report["forbidden_removed_stack_hits"]:
        warnings.append("removed stack terms remain outside negative-results documentation")

    report["status"] = "PASS" if not warnings else "FAIL"
    report["warnings"] = warnings

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Cleanup Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Checks",
        f"- config/candidate.yaml unchanged: {'yes' if not report['candidate_yaml_has_diff'] else 'no'}",
        f"- blocked staged files: {len(report['blocked_staged_files'])}",
        f"- possible sensitive hits: {len(report['sensitive_hits'])}",
        f"- legacy assistant residual files: {len(report['legacy_assistant_residual_files'])}",
        f"- forbidden mainline doc hits: {len(report['forbidden_mainline_doc_hits'])}",
        f"- removed mainline residuals: {len(report['removed_mainline_residuals'])}",
        f"- removed stack text hits: {len(report['forbidden_removed_stack_hits'])}",
        "",
        "## Warnings",
    ]
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("- none")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"clean {report['status']} -> {out / 'report.md'}")
    if warnings:
        raise SystemExit("Cleanup audit failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RNA-OmniPrefold local cleanup state.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    clean = sub.add_parser("clean")
    clean.add_argument("--out", default="outputs/clean")
    clean.set_defaults(func=run_clean)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
