# Significance Check

Generated: 2026-08-28T13:38:08+00:00
Mode: `paired_entry`
Pair key: `entry`
Risk mult: `off`
Paired rows: `1266`
Full paired rows: `1239`
Entry paired rows: `1266`

## Observed

- baseline net: `390.985457`
- candidate net: `416.453308`
- delta net: `25.467851`
- delta R/tr: `0.02011679`

## Bootstrap

- runs: `5000`
- p_gt_zero: `1.0`
- p05_delta_net_r: `17.416181`
- p50_delta_net_r: `25.078442`
- p95_delta_net_r: `33.955422`

## Rule

Treat weak improvements as suspicious when bootstrap lower-tail delta is near
or below zero. For risk-only overlays, full paired mode is expected.
For exit-policy experiments, entry-paired mode is expected because
the same entries can intentionally produce different exits/outcomes.
