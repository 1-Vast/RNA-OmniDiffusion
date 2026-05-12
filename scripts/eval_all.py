"""Eval finetuned deeper model on test + train seeds 123,2024."""
import sys
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

# Eval finetuned model
config, tokenizer, chk = load_checkpoint('outputs/deeper_finetune/best.pt', device)
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

print('Finetuned deeper model:', flush=True)
for ml in [4, 5]:
    preds = []
    for s, l in zip(seqs, all_logits):
        pr = apply_pruning_mask(s, l, 'min_loop_strict', {'min_loop': ml})
        preds.append(nussinov_decode(s, l, min_loop_length=ml, pair_threshold=0.25, nussinov_gamma=2.0, input_is_logit=True, pruning_mask=pr))
    m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
    print(f'  ml={ml}+prune: F1={m["pair_f1"]:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f}', flush=True)

# Compare with original deeper model
config2, tokenizer2, chk2 = load_checkpoint('outputs/deeper/best.pt', device)
model2 = build_model(config2, tokenizer2, device)
model2.load_state_dict(chk2['model_state']); model2.eval()
all_logits2 = []
for seq in seqs:
    b, _, sp = _build_inference_batch(tokenizer2, 'seq2struct', seq, '.'*len(seq), device=device)
    b['input_ids'][:, sp] = tokenizer2.mask_id
    with torch.no_grad(): out = _forward_model(model2, b)
    all_logits2.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())

print('\nOriginal deeper model:', flush=True)
for ml in [4, 5]:
    preds = []
    for s, l in zip(seqs, all_logits2):
        pr = apply_pruning_mask(s, l, 'min_loop_strict', {'min_loop': ml})
        preds.append(nussinov_decode(s, l, min_loop_length=ml, pair_threshold=0.25, nussinov_gamma=2.0, input_is_logit=True, pruning_mask=pr))
    m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
    print(f'  ml={ml}+prune: F1={m["pair_f1"]:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f}', flush=True)

# Ensemble deeper + base models
print('\nEnsemble deeper + base seeds:', flush=True)
base_ckpts = ['outputs/mainline_seed123/best.pt', 'outputs/mainline_seed2024/best.pt']
all_models_logits = [all_logits2]  # deeper seed42
for ckpt_path in base_ckpts:
    config_b, tokenizer_b, chk_b = load_checkpoint(ckpt_path, device)
    model_b = build_model(config_b, tokenizer_b, device)
    model_b.load_state_dict(chk_b['model_state']); model_b.eval()
    logs = []
    for seq in seqs:
        b, _, sp = _build_inference_batch(tokenizer_b, 'seq2struct', seq, '.'*len(seq), device=device)
        b['input_ids'][:, sp] = tokenizer_b.mask_id
        with torch.no_grad(): out = _forward_model(model_b, b)
        logs.append(out['pair_logits'][0, :len(seq), :len(seq)].detach().cpu().numpy())
    all_models_logits.append(logs)

# Ensemble: average logits then decode
for ml in [4, 5]:
    preds = []
    for i in range(len(seqs)):
        L = len(seqs[i])
        ens = np.mean([all_models_logits[m][i][:L,:L] for m in range(3)], axis=0)
        pr = apply_pruning_mask(seqs[i], ens, 'min_loop_strict', {'min_loop': ml})
        preds.append(nussinov_decode(seqs[i], ens, min_loop_length=ml, pair_threshold=0.25, nussinov_gamma=2.0, input_is_logit=True, pruning_mask=pr))
    m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
    print(f'  ensemble ml={ml}+prune: F1={m["pair_f1"]:.4f} Prec={m["pair_precision"]:.4f} Rec={m["pair_recall"]:.4f}', flush=True)

print('\nDONE')
