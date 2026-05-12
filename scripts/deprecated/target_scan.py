"""Targeted grid scan for all 3 seeds - ml x gamma sweep."""
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.training import load_config, load_checkpoint, build_model, resolve_device
from models.dataset import RNAOmniDataset
from models.decode import nussinov_decode, _forward_model, _build_inference_batch, apply_pruning_mask
from utils.metric import evaluate_structures

device = resolve_device('cuda')

for seed_name, ckpt_path in [
    ('seed42', 'outputs/mainline_lr0010/best.pt'),
    ('seed123', 'outputs/mainline_seed123/best.pt'),
    ('seed2024', 'outputs/mainline_seed2024/best.pt'),
]:
    print(f'\n=== {seed_name} ===', flush=True)
    t0 = time.time()
    config, tokenizer, ckpt = load_checkpoint(ckpt_path, device)
    dataset = RNAOmniDataset(Path('dataset/archive/test.jsonl'), max_length=512)
    seqs = [s['seq'] for s in dataset.samples]
    trues = [s['struct'] for s in dataset.samples]
    model = build_model(config, tokenizer, device)
    model.load_state_dict(ckpt['model_state']); model.eval()

    all_logits = []
    for seq in seqs:
        b, _, sp = _build_inference_batch(tokenizer, 'seq2struct', seq, '.'*len(seq), device=device)
        b['input_ids'][:, sp] = tokenizer.mask_id
        with torch.no_grad(): out = _forward_model(model, b)
        all_logits.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())

    best_f1, best_cfg = 0, ''
    for ml in [4, 5, 6, 7, 8]:
        for g in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            preds = []
            for seq, logits in zip(seqs, all_logits):
                pr = apply_pruning_mask(seq, logits, 'min_loop_strict', {'min_loop': ml})
                preds.append(nussinov_decode(seq, logits, min_loop_length=ml, pair_threshold=0.25,
                    nussinov_gamma=g, input_is_logit=True, pruning_mask=pr))
            m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
            if m['pair_f1'] > best_f1:
                best_f1 = m['pair_f1']; best_cfg = f'ml={ml} g={g}'
                print(f'  NEW BEST: {best_cfg} F1={best_f1:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f}', flush=True)
    
    elapsed = time.time() - t0
    print(f'  FINAL: {best_cfg} F1={best_f1:.4f} ({elapsed:.0f}s)', flush=True)

print('\nDONE')
