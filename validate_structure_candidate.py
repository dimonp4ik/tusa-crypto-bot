"""Evaluate the frozen 2026 rule on a corrected older replay, no retuning."""
import json,hashlib
from pathlib import Path
from filter_lab import load,gate,metrics,matches
folder=Path('reports/audit_2026_09_08')
selection=json.loads((folder/'all_structure_filter_search.json').read_text(encoding='utf-8'))
rule=selection['chosen']['rule']
p=folder/'all_structure_2025_raw.csv'
rows=load(p)
report={'frozen_rule':rule,'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'selected':metrics(gate([r for r in rows if matches(r,rule)],3)),
        'baseline':metrics(gate(rows,3)),'status':'RESEARCH_ONLY',
        'limitations':['Retrospective check on an earlier partial-year window, not fresh live evidence.',
        'SWAP prices and modeled costs; funding and actual X-Perp execution excluded.']}
(folder/'all_structure_frozen_2025.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
