"""Test ensemble with correct pruning semantics."""
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.training import load_checkpoint, build_model, resolve_device
from models.dataset import RNAOmniDataset
from models.decode import nussinov_decode, _forward_model, _build_inference_batch, apply_pruning_mask
from utils.metric import evaluate_structures
from utils.struct import parse_dot_bracket

device = resolve_device('cuda')
dataset = RNAOmniDataset(Path('dataset/archive/test.jsonl'), max_length=512)
seqs = [s['seq'] for s in dataset.samples]
trues = [s['struct'] for s in dataset.samples]

# Load all 3 models
ckpts = ['outputs/mainline_lr0010/best.pt', 'outputs/mainline_seed123/best.pt', 'outputs/mainline_seed2024/best.pt']
all_seed_logits = []
for ckpt_path in ckpts:
    config, tokenizer, chk = load_checkpoint(ckpt_path, device)
    model = build_model(config, tokenizer, device)
    model.load_state_dict(chk['model_state']); model.eval()
    logits = []
    for seq in seqs:
        b, _, sp = _build_inference_batch(tokenizer, 'seq2struct', seq, '.'*len(seq), device=device)
        b['input_ids'][:, sp] = tokenizer.mask_id
        with torch.no_grad(): out = _forward_model(model, b)
        logits.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())
    all_seed_logits.append(logits)

N = len(seqs)

def decode_with_prune(seq, logits, ml=4):
    """ml=4 with strict pruning -> effective min_loop=5"""
    pr = apply_pruning_mask(seq, logits, 'min_loop_strict', {'min_loop': ml})
    return nussinov_decode(seq, logits, min_loop_length=ml, pair_threshold=0.25,
        nussinov_gamma=2.0, input_is_logit=True, pruning_mask=pr)

# 1. Single seed42 with ml=4+prune
preds_single = [decode_with_prune(s, l, 4) for s, l in zip(seqs, all_seed_logits[0])]
m = evaluate_structures(preds_single, trues, seqs, allow_wobble=True)
print(f'seed42 ml=4+prune: F1={m["pair_f1"]:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f}', flush=True)

# 2. Ensemble logits then decode
preds_ens = []
for i in range(N):
    L = len(seqs[i])
    ens = np.mean([all_seed_logits[s][i][:L,:L] for s in range(3)], axis=0)
    preds_ens.append(decode_with_prune(seqs[i], ens, 4))
m = evaluate_structures(preds_ens, trues, seqs, allow_wobble=True)
print(f'ensemble+prune:  F1={m["pair_f1"]:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f}', flush=True)

# 3. Per-seed best params (from aggressive_test results)
# seed42 best: ml=4 g=4.0 t=0.25 (effective ml=5)
# seed123 best: ml=6 g=0.5 t=0.10 (effective ml=7)
# seed2024 best: ml=4 g=4.0 t=0.25 (effective ml=5)
per_seed_configs = [(4, 4.0, 0.25), (6, 0.5, 0.10), (4, 4.0, 0.25)]
preds_per_seed = []
for s_idx, (ml, g, t) in enumerate(per_seed_configs):
    preds = []
    for i in range(N):
        L = len(seqs[i])
        l = all_seed_logits[s_idx][i][:L,:L]
        pr = apply_pruning_mask(seqs[i], l, 'min_loop_strict', {'min_loop': ml})
        preds.append(nussinov_decode(seqs[i], l, min_loop_length=ml, pair_threshold=t,
            nussinov_gamma=g, input_is_logit=True, pruning_mask=pr))
    m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
    print(f'seed{s_idx+1} optimal: F1={m["pair_f1"]:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f}', flush=True)
    preds_per_seed.append(preds)

# 4. Per-seed best params, per-sample best (oracle upper bound)
oracle_preds = []
for i in range(N):
    best_f1, best_pred = 0, ''
    for s_idx in range(3):
        f1 = evaluate_structures([preds_per_seed[s_idx][i]], [trues[i]], [seqs[i]], allow_wobble=True)['pair_f1']
        if f1 > best_f1:
            best_f1 = f1; best_pred = preds_per_seed[s_idx][i]
    oracle_preds.append(best_pred)
m = evaluate_structures(oracle_preds, trues, seqs, allow_wobble=True)
print(f'oracle (upper bound): F1={m["pair_f1"]:.4f}', flush=True)

print('\nDONE')
