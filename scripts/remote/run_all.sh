#!/bin/bash
# RNA-OmniPrefold Remote 4090 Experiment Suite
# Run from: /root/autodl-tmp/RNA-OmniDiffusion
# Usage: bash scripts/remote/run_all.sh [d1|d2|d3|d4|all]
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/RNA-OmniDiffusion"
cd "$PROJECT_DIR"

# Colors
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# Environment check
check_env() {
    log "=== Environment Check ==="
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"
    python -m compileall -q models scripts utils main.py && log "compileall OK"
    python main.py smoke && log "smoke OK"
    rm -rf outputs/smoke
}

# ============================================================
# D1: Reproduce mainline 14L/640H/3xPR
# ============================================================
run_d1() {
    log "=== D1: Mainline Reproduction ==="
    mkdir -p outputs/reports
    
    for seed in 42 123 2024; do
        log "Training seed=$seed (500 steps)..."
        python main.py train --config config/remote/d1_mainline.yaml --device cuda --max_steps 500 \
            --train_subset 0 2>/dev/null || true
        
        # Override seed and output
        python -c "
import sys; sys.path.insert(0,'.')
from models.training import train_model, load_config
c = load_config('config/remote/d1_mainline.yaml')
c['training']['seed'] = $seed
c['training']['output_dir'] = 'outputs/d1_seed${seed}'
r = train_model(c, max_steps=500, device_name='cuda')
h = r['history']
bf = max((e.get('val_pair_f1',0) for e in h), default=0)
print(f'D1 seed${seed} BEST_VAL_F1: {bf:.4f}')
"
    done

    log "D1: Evaluating test set with ml pruning..."
    python -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path; import numpy as np; import torch
from models.training import load_checkpoint, build_model, resolve_device
from models.dataset import RNAOmniDataset
from models.decode import nussinov_decode, _forward_model, _build_inference_batch, apply_pruning_mask
from utils.metric import evaluate_structures

device = resolve_device('cuda')
dataset = RNAOmniDataset(Path('dataset/archive/test.jsonl'), max_length=512)
seqs = [s['seq'] for s in dataset.samples]; trues = [s['struct'] for s in dataset.samples]

all_results = []
for seed in [42, 123, 2024]:
    ckpt_path = f'outputs/d1_seed{seed}/best.pt'
    if not Path(ckpt_path).exists():
        print(f'seed{seed}: no checkpoint, skipping')
        continue
    config, tokenizer, chk = load_checkpoint(ckpt_path, device)
    model = build_model(config, tokenizer, device); model.load_state_dict(chk['model_state']); model.eval()
    
    logs = []
    for seq in seqs:
        b, _, sp = _build_inference_batch(tokenizer, 'seq2struct', seq, '.'*len(seq), device=device)
        b['input_ids'][:, sp] = tokenizer.mask_id
        with torch.no_grad(): out = _forward_model(model, b)
        logs.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())
    
    for ml in [4, 5, 6]:
        preds = []
        for s, l in zip(seqs, logs):
            pr = apply_pruning_mask(s, l, 'min_loop_strict', {'min_loop': ml})
            preds.append(nussinov_decode(s, l, min_loop_length=ml, pair_threshold=0.25, nussinov_gamma=2.0, input_is_logit=True, pruning_mask=pr))
        m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
        all_results.append({'seed': seed, 'ml': ml, 'f1': m['pair_f1'], 'prec': m['pair_precision'], 'rec': m['pair_recall']})
        print(f'  seed{seed} ml={ml}: F1={m[\"pair_f1\"]:.4f} Prec={m[\"pair_precision\"]:.4f} Rec={m[\"pair_recall\"]:.4f}')

# Ensemble
all_logits_per_seed = []
for seed in [42, 123, 2024]:
    ckpt_path = f'outputs/d1_seed{seed}/best.pt'
    if not Path(ckpt_path).exists(): continue
    config, tokenizer, chk = load_checkpoint(ckpt_path, device)
    model = build_model(config, tokenizer, device); model.load_state_dict(chk['model_state']); model.eval()
    logs = []
    for seq in seqs:
        b, _, sp = _build_inference_batch(tokenizer, 'seq2struct', seq, '.'*len(seq), device=device)
        b['input_ids'][:, sp] = tokenizer.mask_id
        with torch.no_grad(): out = _forward_model(model, b)
        logs.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())
    all_logits_per_seed.append(logs)

if len(all_logits_per_seed) == 3:
    for ml in [4, 5, 6]:
        preds = []
        for i in range(len(seqs)):
            L = len(seqs[i])
            ens = np.mean([all_logits_per_seed[s][i][:L,:L] for s in range(3)], axis=0)
            pr = apply_pruning_mask(seqs[i], ens, 'min_loop_strict', {'min_loop': ml})
            preds.append(nussinov_decode(seqs[i], ens, min_loop_length=ml, pair_threshold=0.25, nussinov_gamma=2.0, input_is_logit=True, pruning_mask=pr))
        m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
        print(f'  ENSEMBLE ml={ml}: F1={m[\"pair_f1\"]:.4f} Prec={m[\"pair_precision\"]:.4f} Rec={m[\"pair_recall\"]:.4f}')
        all_results.append({'seed': 'ensemble', 'ml': ml, 'f1': m['pair_f1'], 'prec': m['pair_precision'], 'rec': m['pair_recall']})

with open('outputs/reports/d1_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('D1 complete. Results: outputs/reports/d1_results.json')
"

    # Cleanup weights
    find outputs/d1_seed* -name "*.pt" -delete 2>/dev/null || true
    log "D1 done"
}

# ============================================================
# D2: 16L + Axial Pair Attention
# ============================================================
run_d2() {
    log "=== D2: 16L Axial Pair Attention ==="
    
    log "Training axial model (500 steps)..."
    python -c "
import sys; sys.path.insert(0,'.')
from models.training import train_model, load_config
c = load_config('config/remote/d2_axial.yaml')
r = train_model(c, max_steps=500, device_name='cuda')
h = r['history']
bf = max((e.get('val_pair_f1',0) for e in h), default=0)
print(f'D2 BEST_VAL_F1: {bf:.4f}')
"

    log "Evaluating D2 on test..."
    python -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path; import numpy as np; import torch
from models.training import load_checkpoint, build_model, resolve_device
from models.dataset import RNAOmniDataset
from models.decode import nussinov_decode, _forward_model, _build_inference_batch, apply_pruning_mask
from utils.metric import evaluate_structures

device = resolve_device('cuda')
config, tokenizer, chk = load_checkpoint('outputs/d2_axial/best.pt', device)
dataset = RNAOmniDataset(Path('dataset/archive/test.jsonl'), max_length=512)
seqs = [s['seq'] for s in dataset.samples]; trues = [s['struct'] for s in dataset.samples]
model = build_model(config, tokenizer, device); model.load_state_dict(chk['model_state']); model.eval()

logs = []
for seq in seqs:
    b, _, sp = _build_inference_batch(tokenizer, 'seq2struct', seq, '.'*len(seq), device=device)
    b['input_ids'][:, sp] = tokenizer.mask_id
    with torch.no_grad(): out = _forward_model(model, b)
    logs.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())

for ml in [4, 5, 6]:
    preds = []
    for s, l in zip(seqs, logs):
        pr = apply_pruning_mask(s, l, 'min_loop_strict', {'min_loop': ml})
        preds.append(nussinov_decode(s, l, min_loop_length=ml, pair_threshold=0.25, nussinov_gamma=2.0, input_is_logit=True, pruning_mask=pr))
    m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
    print(f'  D2 axial ml={ml}: F1={m[\"pair_f1\"]:.4f} Prec={m[\"pair_precision\"]:.4f} Rec={m[\"pair_recall\"]:.4f}')
" | tee outputs/reports/d2_results.txt

    find outputs/d2_axial -name "*.pt" -delete 2>/dev/null || true
    log "D2 done"
}

# ============================================================
# D3: 3-seed Ensemble
# ============================================================
run_d3() {
    log "=== D3: Multi-seed Ensemble (runs inside D1) ==="
    log "D3 results are included in D1 ensemble section"
}

# ============================================================
# D4: Bilinear pair head
# ============================================================
run_d4() {
    log "=== D4: Bilinear Pair Head ==="
    
    python -c "
import sys; sys.path.insert(0,'.')
from models.training import train_model, load_config
c = load_config('config/remote/d4_bilinear.yaml')
r = train_model(c, max_steps=500, device_name='cuda')
h = r['history']
bf = max((e.get('val_pair_f1',0) for e in h), default=0)
print(f'D4 BEST_VAL_F1: {bf:.4f}')
"

    python -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path; import numpy as np; import torch
from models.training import load_checkpoint, build_model, resolve_device
from models.dataset import RNAOmniDataset
from models.decode import nussinov_decode, _forward_model, _build_inference_batch, apply_pruning_mask
from utils.metric import evaluate_structures

device = resolve_device('cuda')
config, tokenizer, chk = load_checkpoint('outputs/d4_bilinear/best.pt', device)
dataset = RNAOmniDataset(Path('dataset/archive/test.jsonl'), max_length=512)
seqs = [s['seq'] for s in dataset.samples]; trues = [s['struct'] for s in dataset.samples]
model = build_model(config, tokenizer, device); model.load_state_dict(chk['model_state']); model.eval()

logs = []
for seq in seqs:
    b, _, sp = _build_inference_batch(tokenizer, 'seq2struct', seq, '.'*len(seq), device=device)
    b['input_ids'][:, sp] = tokenizer.mask_id
    with torch.no_grad(): out = _forward_model(model, b)
    logs.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())

for ml in [4, 5, 6]:
    preds = []
    for s, l in zip(seqs, logs):
        pr = apply_pruning_mask(s, l, 'min_loop_strict', {'min_loop': ml})
        preds.append(nussinov_decode(s, l, min_loop_length=ml, pair_threshold=0.25, nussinov_gamma=2.0, input_is_logit=True, pruning_mask=pr))
    m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
    print(f'  D4 bilinear ml={ml}: F1={m[\"pair_f1\"]:.4f} Prec={m[\"pair_precision\"]:.4f} Rec={m[\"pair_recall\"]:.4f}')
" | tee outputs/reports/d4_results.txt

    find outputs/d4_bilinear -name "*.pt" -delete 2>/dev/null || true
    log "D4 done"
}

# ============================================================
# ViennaRNA baseline
# ============================================================
run_vienna() {
    log "=== ViennaRNA Baseline ==="
    python -c "
import subprocess, json
from pathlib import Path
from utils.metric import evaluate_structures
from utils.struct import parse_dot_bracket

dataset = []
with open('dataset/archive/test.jsonl') as f:
    for line in f:
        if line.strip(): dataset.append(json.loads(line))

seqs = [s['seq'] for s in dataset]
trues = [s['struct'] for s in dataset]
preds = []

for i, s in enumerate(seqs):
    result = subprocess.run(['RNAfold', '--noPS'], input=s, capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    struct = lines[1].split()[0] if len(lines) > 1 else '.' * len(s)
    preds.append(struct)
    if (i+1) % 50 == 0:
        print(f'  RNAfold: {i+1}/{len(seqs)}')

m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
print(f'ViennaRNA F1={m[\"pair_f1\"]:.4f} Prec={m[\"pair_precision\"]:.4f} Rec={m[\"pair_recall\"]:.4f}')
print(f'Valid={m[\"valid_structure_rate\"]:.3f}')
with open('outputs/reports/vienna_baseline.json', 'w') as f:
    json.dump({'f1': m['pair_f1'], 'prec': m['pair_precision'], 'rec': m['pair_recall'], 'valid': m['valid_structure_rate']}, f)
" | tee outputs/reports/vienna_baseline.txt
}

# ============================================================
# Cleanup
# ============================================================
cleanup() {
    log "=== Cleanup ==="
    find outputs -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" | xargs rm -f 2>/dev/null || true
    find outputs -name "predictions.jsonl" -delete 2>/dev/null || true
    du -sh outputs/
    log "Cleanup done. Reports: outputs/reports/"
}

# ============================================================
# Main
# ============================================================
mkdir -p outputs/reports

case "${1:-all}" in
    check)  check_env ;;
    d1)     check_env && run_d1 && cleanup ;;
    d2)     check_env && run_d2 && cleanup ;;
    d3)     run_d3 ;;
    d4)     check_env && run_d4 && cleanup ;;
    vienna) run_vienna ;;
    all)    check_env && run_vienna && run_d1 && run_d2 && run_d4 && cleanup ;;
    *)      echo "Usage: $0 {check|d1|d2|d3|d4|vienna|all}" ;;
esac

log "All experiments complete!"
