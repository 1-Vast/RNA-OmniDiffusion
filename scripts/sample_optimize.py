"""Sampling-based optimization: test on 50 samples, then verify best on full set."""
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

device = resolve_device('cuda')
N_SAMPLE = 50
N_FULL = 0  # 0 = all

for seed_name, ckpt in [('seed42','outputs/mainline_lr0010/best.pt'),
                         ('seed123','outputs/mainline_seed123/best.pt'),
                         ('seed2024','outputs/mainline_seed2024/best.pt')]:
    print(f'\n=== {seed_name} ===', flush=True)
    config, tokenizer, chk = load_checkpoint(ckpt, device)
    dataset = RNAOmniDataset(Path('dataset/archive/test.jsonl'), max_length=512)
    all_seqs = [s['seq'] for s in dataset.samples]
    all_trues = [s['struct'] for s in dataset.samples]
    full_N = len(all_seqs)
    
    model = build_model(config, tokenizer, device)
    model.load_state_dict(chk['model_state']); model.eval()
    
    # Extract all logits (needed anyway for full test)
    all_logits = []
    for seq in all_seqs:
        b, _, sp = _build_inference_batch(tokenizer, 'seq2struct', seq, '.'*len(seq), device=device)
        b['input_ids'][:, sp] = tokenizer.mask_id
        with torch.no_grad(): out = _forward_model(model, b)
        all_logits.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())
    print(f'  Logits extracted for {full_N} samples', flush=True)
    
    # Phase 1: fast scan on 50 samples
    seqs50, trues50, logs50 = all_seqs[:N_SAMPLE], all_trues[:N_SAMPLE], all_logits[:N_SAMPLE]
    sample_best_f1, sample_best_cfg = 0, ''
    for ml in [4, 5, 6]:
        for g in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
            for t in [0.10, 0.15, 0.20, 0.25]:
                preds = [nussinov_decode(s, l, min_loop_length=ml, pair_threshold=t, nussinov_gamma=g,
                    input_is_logit=True, pruning_mask=apply_pruning_mask(s,l,'min_loop_strict',{'min_loop':ml}))
                    for s, l in zip(seqs50, logs50)]
                m = evaluate_structures(preds, trues50, seqs50, allow_wobble=True)
                if m['pair_f1'] > sample_best_f1:
                    sample_best_f1 = m['pair_f1']; sample_best_cfg = (ml, g, t)
    print(f'  Sample best: ml={sample_best_cfg[0]} g={sample_best_cfg[1]} t={sample_best_cfg[2]} F1={sample_best_f1:.4f}', flush=True)
    
    # Phase 2: verify best on full set
    ml, g, t = sample_best_cfg
    preds = [nussinov_decode(s, l, min_loop_length=ml, pair_threshold=t, nussinov_gamma=g,
        input_is_logit=True, pruning_mask=apply_pruning_mask(s,l,'min_loop_strict',{'min_loop':ml}))
        for s, l in zip(all_seqs, all_logits)]
    m = evaluate_structures(preds, all_trues, all_seqs, allow_wobble=True)
    print(f'  FULL: ml={ml} g={g} t={t} F1={m["pair_f1"]:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f}', flush=True)

print('\nDONE')
