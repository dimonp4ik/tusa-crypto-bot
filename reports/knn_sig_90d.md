# Significance Check

Generated: 2026-06-13T17:16:37+00:00
Mode: `paired`
Pair key: `full`
Risk mult: `on`
Paired rows: `659`
Full paired rows: `659`
Entry paired rows: `659`

## Observed

- baseline net: `311.832104`
- candidate net: `329.164522`
- delta net: `17.332417`
- delta R/tr: `0.02630109`

## Bootstrap

- runs: `10000`
- p_gt_zero: `0.9998`
- p05_delta_net_r: `9.288463`
- p50_delta_net_r: `17.388273`
- p95_delta_net_r: `25.493234`

## Rule

Treat weak improvements as suspicious when bootstrap lower-tail delta is near
or below zero. For risk-only overlays, full paired mode is expected.
For exit-policy experiments, entry-paired mode is expected because
the same entries can intentionally produce different exits/outcomes.
