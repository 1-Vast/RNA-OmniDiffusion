# Pair Calibration Analysis

## Global pair_logit_offset

### Experiment: offset=-2.0, multi-seed

| Seed  | Val F1 | Test F1 | Precision | Recall | Overpair |
|-------|--------|---------|-----------|--------|----------|
| 42    | 0.388  | 0.382   | -         | -      | down     |
| 123   | crash  | crash   | crash     | crash  | -        |
| 2024  | drop   | drop    | -         | -      | -        |

Multi-seed mean F1: 0.316 ± 0.076 (worse than mainline 0.351 ± 0.028)

### Conclusion
- **Global pair_logit_offset is NOT mainline.**
- Single-seed (seed42) positive, but multi-seed negative with catastrophic seed123 failure.
- Reason: seed-dependent recall collapse.
- `pair_logit_offset` retained as optional tuning knob, default must be 0.0.
- Current mainline: `pair_logit_offset: 0.0`, `pair_calibration.enabled: false`.

### Next Steps
Move to structure-aware, localized pair calibration:
- Candidate pruning (canonical, loop, distance)
- Energy-inspired pair prior
- Pair-specific bias table
- Local pair ranking loss
