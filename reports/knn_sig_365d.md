# Significance Check

Generated: 2026-06-13T17:22:18+00:00
Mode: `paired`
Pair key: `full`
Risk mult: `on`
Paired rows: `1888`
Full paired rows: `1888`
Entry paired rows: `1888`

## Observed

- baseline net: `879.008831`
- candidate net: `904.623599`
- delta net: `25.614768`
- delta R/tr: `0.01356714`

## Bootstrap

- runs: `10000`
- p_gt_zero: `0.999`
- p05_delta_net_r: `12.350653`
- p50_delta_net_r: `25.637552`
- p95_delta_net_r: `39.189195`

## Rule

Treat weak improvements as suspicious when bootstrap lower-tail delta is near
or below zero. For risk-only overlays, full paired mode is expected.
For exit-policy experiments, entry-paired mode is expected because
the same entries can intentionally produce different exits/outcomes.
