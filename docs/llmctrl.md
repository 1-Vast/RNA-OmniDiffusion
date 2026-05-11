# LLM-Ctrl: LLM as Training Controller

## Architecture

LLM-Ctrl redesigns LLM's role from preference judge to training controller.
LLM analyzes validation error profiles and proposes MS-MPRM relation-masking
policies. It never enters model forward or generates structure labels.

```
Error Profiler → LLM Strategy Planner → Policy Validator → MS-MPRM Training
     ↑                                                        ↓
     └────────────── validation feedback ←───────────────────┘
```

## Components

### 1. Error Profiler
Reads validation benchmark output, identifies top error types:
over-pairing, under-pairing, broken stems, long-range failure, all-dot.

### 2. LLM Strategy Planner
Takes error profile + hard-case examples, outputs 8-12 policy proposals
with different relation-masking ratios targeting specific error types.

### 3. Policy Validator
Validates JSON fields, clips out-of-range values, falls back to rule policy
if LLM output is invalid.

### 4. Curriculum Scheduler (Future)
Plans which policy to apply at each training stage.

## What LLM Is Allowed To Output

- policy_name: string identifier
- relation_mask: {total_ratio, global_ratio, stem_span_ratio, hard_negative_ratio, stem_span_len, long_range_threshold}
- decode_hint: {gamma_candidates, threshold_candidates}
- expected_effect: string describing expected impact
- risk: string describing potential downside
- confidence: 0-1

## What LLM Is NOT Allowed To Output

- dot-bracket structures
- base-pair labels (i,j) pairs
- model forward tokens
- direct structure predictions
- free-form training code
- unvalidated parameters

## Required Controls

Every LLM policy must be compared against:
1. fixed MS-MPRM (baseline relation config)
2. rule policy (heuristic error→policy mapping)
3. random policy (sampled from valid action space)
4. shuffled policy (LLM policies with shuffled parameter values)

## Criteria for LLM Contribution

LLM policy is considered contributive only if ALL of:
1. best LLM policy > fixed MS-MPRM
2. best LLM policy > best rule policy
3. best LLM policy > best random policy
4. Pair Ratio does not worsen
5. Valid rate does not drop

## Usage

```bash
# Generate error profile
python scripts/eval.py profile --config config/msmprm.yaml --ckpt outputs/msmprm/best.pt --out outputs/ctrl/profile.json

# Propose policies (LLM)
python scripts/coach.py --mode propose --profile outputs/ctrl/profile.json --out outputs/ctrl/llm_policies.jsonl --n 12

# Propose policies (rule)
python scripts/coach.py --mode proposerule --profile outputs/ctrl/profile.json --out outputs/ctrl/rule_policies.jsonl

# Materialize configs
python scripts/coach.py --mode materialize --proposals outputs/ctrl/llm_policies.jsonl --template config/msmprm.yaml --outdir config/ctrl_llm
```
