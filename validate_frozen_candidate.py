"""Validate one frozen candidate without retuning thresholds on older windows."""
import json,hashlib
from pathlib import Path
from filter_lab import load,gate,metrics,matches
folder=Path('reports/audit_2026_09_08')
selection=json.loads((folder/'filter_search_taker.json').read_text())
if not selection['chosen']:raise SystemExit('No selected rule')
rule=selection['chosen']['rule'];results=[]
for label in ['filter_2023','filter_2024','filter_universe']:
 p=folder/(label+'_raw.csv');meta=json.loads((folder/(label+'.json')).read_text())
 old=2*(meta['arguments']['fee_rate']+meta['arguments']['slippage_rate']);rows=load(p)
 for r in rows:
  cost=r['cost_r'];r['net_r']+=cost-cost*.0012/old;r['cost_r']=cost*.0012/old
 selected=gate([r for r in rows if matches(r,rule)],3)
 results.append({'window':label,'selected':metrics(selected),'baseline':metrics(gate(rows,3)),
                 'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
report={'status':'RESEARCH_ONLY','frozen_rule':rule,'results':results,
 'limitations':['Retrospective robustness check; 2026 has been inspected in previous searches.',
 'Historical windows cover parts of years, not full calendar years. Funding omitted.',
 'SWAP signals/exits and legacy risk weights; actual X-Perp profitability not established.']}
(folder/'frozen_taker_candidate_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
