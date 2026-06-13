# Significance Check

Generated: 2026-06-13T17:18:40+00:00
Mode: `paired`
Pair key: `full`
Risk mult: `on`
Paired rows: `880`
Full paired rows: `880`
Entry paired rows: `880`

## Observed

- baseline net: `449.195724`
- candidate net: `470.498841`
- delta net: `21.303116`
- delta R/tr: `0.02420809`

## Bootstrap

- runs: `10000`
- p_gt_zero: `0.9999`
- p05_delta_net_r: `12.314774`
- p50_delta_net_r: `21.409542`
- p95_delta_net_r: `30.557859`

## Rule

Treat weak improvements as suspicious when bootstrap lower-tail delta is near
or below zero. For risk-only overlays, full paired mode is expected.
For exit-policy experiments, entry-paired mode is expected because
the same entries can intentionally produce different exits/outcomes.
