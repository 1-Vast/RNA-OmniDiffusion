"""One-shot data processing: bpRNA + RNAcentral for RNA-OmniPrefold experiments."""
import gzip, json, random, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def clean_seq(seq):
    """Uppercase, T->U, filter non-AUCGN to N."""
    seq = seq.upper().replace("T", "U")
    return "".join(c if c in "AUCGN" else "N" for c in seq)


def parse_dbn(filepath):
    """Parse a .dbn file: returns (name, sequence, dot_bracket_structure)."""
    lines = Path(filepath).read_text().splitlines()
    name = None
    seq = None
    struct = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("#"):
            continue
        elif seq is None:
            # First non-comment line should be the sequence
            # Check it's mostly AUCG characters
            if sum(1 for c in line.upper() if c in "AUCGN") > len(line) * 0.7:
                seq = line
        elif struct is None:
            # Second non-comment line should be dot-bracket structure
            if sum(1 for c in line if c in ".()[]{}<>") > len(line) * 0.7:
                raw_struct = "".join(c for c in line if c in ".()[]{}<>")
                if len(raw_struct) == len(seq):
                    struct = raw_struct
    return name, seq, struct


def pairs_from_dot_bracket(struct):
    """Convert dot-bracket string to list of (i,j) pairs."""
    pairs = []
    stack = []
    for i, c in enumerate(struct):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            pairs.append((j, i))
    return pairs


def process_bprna():
    """Process bpRNA-1m(90): FASTA + dbn -> struct JSONL + seq JSONL."""
    print("=" * 50)
    print("Processing bpRNA-1m(90)")
    print("=" * 50)

    fasta_path = REPO / "dataset/raw/bprna_1m90/bpRNA_1m_90.fasta"
    dbn_dir = REPO / "dataset/raw/bprna_1m90/bpRNA_1m_90_DBNFILES"
    out_dir = REPO / "dataset/processed/bprna"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build dbn lookup
    print("Loading dbn files...")
    dbn_map = {}
    dbn_files = sorted(dbn_dir.glob("*.dbn"))
    for i, dbn_file in enumerate(dbn_files):
        name, seq, struct = parse_dbn(dbn_file)
        if name and seq and struct:
            dbn_map[name] = (clean_seq(seq), struct)
        if (i + 1) % 5000 == 0:
            print(f"  dbn: {i+1}/{len(dbn_files)} loaded={len(dbn_map)}")
    print(f"  dbn loaded: {len(dbn_map)}")

    # Parse FASTA
    print("Parsing FASTA and matching with dbn...")
    struct_rows = []
    seq_rows = []
    matched = 0
    unmatched = 0

    with open(fasta_path) as fh:
        sid = None
        sseq = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid and sseq:
                    full_seq = clean_seq("".join(sseq))
                    length = len(full_seq)
                    if 20 <= length <= 512:
                        if sid in dbn_map:
                            dbn_seq, dbn_struct = dbn_map[sid]
                            pairs = pairs_from_dot_bracket(dbn_struct)
                            struct_rows.append({
                                "id": sid,
                                "seq": full_seq,
                                "struct": dbn_struct,
                                "pairs": pairs,
                                "length": length,
                                "source": "bprna_1m90"
                            })
                            seq_rows.append({
                                "id": sid,
                                "seq": full_seq,
                                "source": "bprna_1m90"
                            })
                            matched += 1
                        else:
                            unmatched += 1
                sid = line[1:].split()[0]
                sseq = []
            else:
                sseq.append(line)

    print(f"  matched: {matched}, unmatched: {unmatched}")

    # Split struct data: 80/10/10
    rng = random.Random(42)
    rng.shuffle(struct_rows)
    n = len(struct_rows)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = [
        ("train", struct_rows[:train_end]),
        ("val", struct_rows[train_end:val_end]),
        ("test", struct_rows[val_end:]),
    ]
    for name, rows in splits:
        path = out_dir / f"bprna_1m90_{name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {name}: {len(rows)} samples")

    # Seq-only (all matched)
    rng.shuffle(seq_rows)
    with open(out_dir / "bprna_1m90_seq.jsonl", "w") as f:
        for r in seq_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  seq-only total: {len(seq_rows)} samples")

    # Sample subsets for pretraining
    for limit in [10000, 50000]:
        if limit <= len(seq_rows):
            sampled = rng.sample(seq_rows, limit)
            path = out_dir / f"bprna_1m90_seq_{limit//1000}k.jsonl"
            with open(path, "w") as f:
                for r in sampled:
                    f.write(json.dumps(r) + "\n")
            print(f"  seq-{limit//1000}k: {len(sampled)} samples")

    return struct_rows, seq_rows


def process_rnacentral(limit=50000):
    """Stream RNAcentral active FASTA and sample sequences."""
    print("\n" + "=" * 50)
    print(f"Processing RNAcentral active (sampling {limit})")
    print("=" * 50)

    rnac_path = REPO / "dataset/RNAcentral/rnacentral_active.fasta.gz"
    out_dir = REPO / "dataset/processed/rnacentral"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    # First pass: collect candidates (reservoir-like)
    print("Streaming FASTA...")
    candidates = []
    seen = set()

    with gzip.open(rnac_path, "rt", encoding="utf-8", errors="replace") as fh:
        sid = None
        sseq = []
        count = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid and sseq:
                    seq = clean_seq("".join(sseq))
                    length = len(seq)
                    if 20 <= length <= 512 and seq not in seen:
                        seen.add(seq)
                        candidates.append({
                            "id": sid,
                            "seq": seq,
                            "source": "rnacentral_active"
                        })
                    count += 1
                    if count % 100000 == 0:
                        print(f"  scanned: {count}, candidates: {len(candidates)}")
                sid = line[1:].split()[0]
                sseq = []
            else:
                sseq.append(line)
            if len(candidates) >= limit * 3:
                break  # Enough candidates for sampling

    print(f"  total scanned: {count}, candidates: {len(candidates)}")

    # Sample target number
    sampled = rng.sample(candidates, min(limit, len(candidates)))

    out_path = out_dir / f"rnacentral_active_{limit//1000}k.jsonl"
    with open(out_path, "w") as f:
        for r in sampled:
            f.write(json.dumps(r) + "\n")
    print(f"  rnacentral_{limit//1000}k: {len(sampled)} sequences")
    print(f"  output: {out_path}")

    return sampled


if __name__ == "__main__":
    print("RNA-OmniPrefold Data Processing Pipeline")
    print(f"REPO: {REPO}")
    print()

    # Process bpRNA
    struct_rows, seq_rows = process_bprna()

    # Process RNAcentral (50k + 100k)
    process_rnacentral(limit=50000)
    process_rnacentral(limit=100000)

    print("\n" + "=" * 50)
    print("ALL DATA PROCESSING COMPLETE")
    print("=" * 50)
