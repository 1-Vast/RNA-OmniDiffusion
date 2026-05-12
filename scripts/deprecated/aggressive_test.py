"""Test aggressive decode params for overpair reduction."""
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

# Test all 3 seeds with aggressive params
for ckpt in ['outputs/mainline_lr0010/best.pt', 'outputs/mainline_seed123/best.pt', 'outputs/mainline_seed2024/best.pt']:
    config, tokenizer, chk = load_checkpoint(ckpt, device)
    dataset = RNAOmniDataset(Path('dataset/archive/test.jsonl'), max_length=512)
    seqs = [s['seq'] for s in dataset.samples]
    trues = [s['struct'] for s in dataset.samples]
    model = build_model(config, tokenizer, device)
    model.load_state_dict(chk['model_state']); model.eval()
    
    all_logits = []
    for seq in seqs:
        b, _, sp = _build_inference_batch(tokenizer, 'seq2struct', seq, '.'*len(seq), device=device)
        b['input_ids'][:, sp] = tokenizer.mask_id
        with torch.no_grad(): out = _forward_model(model, b)
        all_logits.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())
    
    # Test aggressive combos
    combos = [
        ('baseline', 3, 2.0, 0.25),
        ('ml7 g2.0 t0.25', 7, 2.0, 0.25),
        ('ml8 g2.0 t0.25', 8, 2.0, 0.25),
        ('ml4 g2.0 t0.40', 4, 2.0, 0.40),
        ('ml4 g2.0 t0.50', 4, 2.0, 0.50),
        ('ml4 g1.0 t0.35', 4, 1.0, 0.35),
        ('ml4 g4.0 t0.25', 4, 4.0, 0.25),
        ('ml5 g2.0 t0.30', 5, 2.0, 0.30),
        ('ml6 g2.0 t0.30', 6, 2.0, 0.30),
        ('ml6 g0.5 t0.10', 6, 0.5, 0.10),
    ]
    
    seed = ckpt.split('_')[-1].replace('/best.pt','')
    print(f'\n{seed}:', flush=True)
    for name, ml, g, t in combos:
        preds = []
        for s, l in zip(seqs, all_logits):
            pr = apply_pruning_mask(s, l, 'min_loop_strict', {'min_loop': ml}) if ml > 3 else None
            preds.append(nussinov_decode(s, l, min_loop_length=max(3,ml), pair_threshold=t,
                nussinov_gamma=g, input_is_logit=True, pruning_mask=pr))
        m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
        op = sum(1 for p, t in zip(preds, trues) if len(parse_dot_bracket(p)) > len(parse_dot_bracket(t))) / len(preds)
        print(f'  {name}: F1={m["pair_f1"]:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f} Overpair={op:.2f}', flush=True)

print('\nDONE')
