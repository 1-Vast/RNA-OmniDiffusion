"""Test small negative offsets with ml=4 - balanced approach."""
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
dataset = RNAOmniDataset(Path('dataset/archive/test.jsonl'), max_length=512)
seqs = [s['seq'] for s in dataset.samples]
trues = [s['struct'] for s in dataset.samples]

for ckpt_path in ['outputs/mainline_lr0010/best.pt', 'outputs/mainline_seed123/best.pt', 'outputs/mainline_seed2024/best.pt']:
    name = ckpt_path.split('/')[-2]
    config, tokenizer, chk = load_checkpoint(ckpt_path, device)
    model = build_model(config, tokenizer, device)
    model.load_state_dict(chk['model_state']); model.eval()
    
    all_logits = []
    for seq in seqs:
        b, _, sp = _build_inference_batch(tokenizer, 'seq2struct', seq, '.'*len(seq), device=device)
        b['input_ids'][:, sp] = tokenizer.mask_id
        with torch.no_grad(): out = _forward_model(model, b)
        all_logits.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())
    
    print(f'\n{name}:', flush=True)
    for offset in [0.0, -0.2, -0.5, -0.75, -1.0, -1.5]:
        for ml in [4, 5]:
            preds = []
            for s, l in zip(seqs, all_logits):
                shifted = l.copy() + offset
                pr = apply_pruning_mask(s, shifted, 'min_loop_strict', {'min_loop': ml})
                preds.append(nussinov_decode(s, shifted, min_loop_length=ml, pair_threshold=0.25,
                    nussinov_gamma=2.0, input_is_logit=True, pruning_mask=pr))
            m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
            print(f'  off={offset:+4.1f} ml={ml}: F1={m["pair_f1"]:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f}', flush=True)
