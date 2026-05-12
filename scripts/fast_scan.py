"""Ultra-fast scan: only most promising combos for each seed."""
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

for seed_name, ckpt in [('seed42','outputs/mainline_lr0010/best.pt'),
                         ('seed123','outputs/mainline_seed123/best.pt'),
                         ('seed2024','outputs/mainline_seed2024/best.pt')]:
    config, tokenizer, chk = load_checkpoint(ckpt, device)
    dataset = RNAOmniDataset(Path('dataset/archive/test.jsonl'), max_length=512)
    seqs = [s['seq'] for s in dataset.samples]
    trues = [s['struct'] for s in dataset.samples]
    model = build_model(config, tokenizer, device)
    model.load_state_dict(chk['model_state']); model.eval()
    
    # Fast logit extraction
    all_logits = []
    for seq in seqs:
        b, _, sp = _build_inference_batch(tokenizer, 'seq2struct', seq, '.'*len(seq), device=device)
        b['input_ids'][:, sp] = tokenizer.mask_id
        with torch.no_grad(): out = _forward_model(model, b)
        all_logits.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())
    
    # Key combos: ml=4,5 with best gamma/threshold ranges
    combos = []
    for ml in [4, 5, 6]:
        for g in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
            for t in [0.10, 0.15, 0.20, 0.25]:
                combos.append((ml, g, t))
    
    best_f1, best_cfg = 0, ''
    for ml, g, t in combos:
        preds = []
        for seq, logits in zip(seqs, all_logits):
            pr = apply_pruning_mask(seq, logits, 'min_loop_strict', {'min_loop': ml})
            preds.append(nussinov_decode(seq, logits, min_loop_length=ml, pair_threshold=t,
                nussinov_gamma=g, input_is_logit=True, pruning_mask=pr))
        m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
        if m['pair_f1'] > best_f1:
            best_f1 = m['pair_f1']; best_cfg = f'ml={ml} g={g} t={t}'
    print(f'{seed_name}: BEST {best_cfg} F1={best_f1:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f}', flush=True)

print('DONE')
