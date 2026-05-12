# Remote 4090 Experiment Plan

## Target
Run RNA-OmniPrefold on remote 4090 server to push F1 toward 0.5.

## Remote Server
- Path: `/root/autodl-tmp/RNA-OmniDiffusion`
- Dataset: `/root/autodl-tmp/RNA-OmniDiffusion/dataset` (MUST preserve)
- GPU: NVIDIA RTX 4090 (24GB VRAM)

## Setup Steps

### 1. Clean old project
```bash
cd /root/autodl-tmp/RNA-OmniDiffusion
# Keep: dataset, .env
# Delete: outputs, checkpoints, old code
find . -maxdepth 1 ! -name dataset ! -name .env -exec rm -rf {} +
```

### 2. Sync clean mainline from local
```bash
# From local Windows:
rsync -avz --exclude='.env' --exclude='outputs/' --exclude='dataset/' \
  README.md INDEX.md main.py models/ scripts/ utils/ config/ docs/ \
  user@remote:/root/autodl-tmp/RNA-OmniDiffusion/
```

### 3. Verify environment
```bash
conda activate DL
python -c "import torch; print(torch.cuda.get_device_name(0))"
RNAfold --version
```

### 4. Smoke test
```bash
python main.py smoke
python -m compileall -q models scripts utils main.py
```

## Experiment Plan

### D1: Reproduce mainline (14L/640H/3xPR)
```bash
python main.py train --config config/mainline_strongest.yaml --device cuda --max_steps 1000
python scripts/eval.py bench --config config/mainline_strongest.yaml --ckpt outputs/mainline_strongest/best.pt --split test --device cuda --decode nussinov --stage_logits
```
Seed 42, 123, 2024. Target: F1 ≥ 0.46.

### D2: 16L + Axial Pair Attention
Config: `config/axial.yaml` (16L, 640H, axial pairrefine)
Target: F1 ≥ 0.48 (seed42).

### D3: 3x Scaled14L Ensemble
Train 3 seeds, average pair logits before decode.
Target: F1 ≥ 0.48-0.50.

### D4: Outer-product pair head
Test bilinear/low-rank outer product head.
Target: F1 improvement ≥ 0.01.

## Post-experiment cleanup
```bash
# Delete checkpoints, keep reports
find outputs -type f \( -name "*.pt" -o -name "*.pth" \) -delete
# Delete large prediction files
find outputs -name "predictions.jsonl" -delete
```

## Success Criteria
- 3-seed mean F1 ≥ 0.48 on same dataset/split
- ViennaRNA rerun on same data for comparison
- All modules LLM-free
- No global pair_logit_offset in defaults
