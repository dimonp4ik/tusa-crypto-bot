"""Compact selection-only review; paper diagnostics are never client cash PnL."""
from bisect import bisect_left
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import math
import pandas as pd

from stock_session_joint_nov19_audit import route as base_route,paper_portfolio,ROUTES as BASE_ROUTES
from stock_session_extended_joint_nov19_audit import route as extended_route,ROUTES as EXTENDED_ROUTES
from phase_scenarios_nov19_audit import ROOT
from account_cash_guard_nov17 import stamp


def readiness_reasons(frame):
    closed=frame.sort_values(['exit_time','entry_time','symbol']).to_dict('records')
    times=[r['exit_time'] for r in closed];counts=Counter()
    for row in frame.to_dict('records'):
        at=bisect_left(times,row['entry_time']);past=closed[max(0,at-20):at]
        if len(past)<20:counts['insufficient20_completed']+=1;continue
        positive_r=math.fsum(r['net_r'] for r in past)>1e-10
        positive_price=math.fsum(r['signed_return'] for r in past)>1e-12
        counts['ready' if positive_r and positive_price else 'paper_r_nonpositive' if not positive_r else 'paper_price_nonpositive']+=1
    return dict(counts)


def main():
    names=('stock_session_joint_nov19','stock_session_extended_joint_nov19')
    paths=[Path(__file__),Path('stock_session_joint_nov19_audit.py'),Path('stock_session_extended_joint_nov19_audit.py')]
    paths.extend(ROOT/f'{name}.json' for name in names)
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};results={}
    for name in names:
        report=json.loads((ROOT/f'{name}.json').read_text(encoding='utf-8'))
        cases={};routing=base_route if name==names[0] else extended_route
        routes=BASE_ROUTES if name==names[0] else EXTENDED_ROUTES
        for kind in ('trade','mark'):
            path=ROOT/f'{name}_{kind}.pkl';raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if report['sources'][str(path)]!=digest:raise ValueError('session_review_cache_changed')
            sources[str(path)]=digest;frame=pd.read_pickle(path)
            for universe in ('base','expanded_current_snapshot'):
                for label in routes:
                    key='|'.join((universe,kind,label));selection=report['selection'][key]
                    if selection['status']!='RESOLVED_RESEARCH_CASH':
                        cases[key]=dict(status=selection['status']);continue
                    sf=routing(frame,label,universe,stamp('2026-08-01'),stamp('2026-10-01'))
                    paper=paper_portfolio(sf);months={}
                    for month in ('2026-08','2026-09'):
                        start=stamp(month+'-01');end=stamp(f'2026-{int(month[-2:])+1:02}-01')
                        rows=paper[paper.exit_time.ge(start)&paper.exit_time.lt(end)]
                        months[month]=dict(paper_closed=len(rows),paper_sum_r=float(rows.net_r.sum()),
                            paper_sum_price_return=float(rows.signed_return.sum()),
                            paper_profitable_exit_pct=100*float(rows.signed_return.gt(0).mean()) if len(rows) else None)
                    # The original generic calendar helper emits later zero
                    # months even though no later entries were passed. Restrict
                    # the review to declared selection months; zeros are not OOS.
                    selected_cash={p:{q:{m:v for m,v in item['calendar_months'].items()
                        if m in ('2026-08','2026-09')} for q,item in v.items()} for p,v in selection['cash'].items()}
                    cases[key]=dict(status=selection['status'],paper_signals=len(paper),
                        readiness_reasons=readiness_reasons(paper),paper_months=months,selection_cash=selected_cash,
                        stock_qualified=selection['stock_qualified'],joint_qualified=selection['joint_qualified'])
        results[name]=dict(signals=report['signal_counts'],selection_cases=cases,
            selected_stock=report['chosen_stock'],later_evaluated=report['later'] is not None)
        k='expanded_current_snapshot|trade|session_union';print(name,'union',
            {z:results[name]['selection_cases'][k].get(z) for z in ('paper_signals','readiness_reasons','paper_months')},flush=True)
    review=dict(status='SELECTION_ONLY_PAPER_AND_SHARED_CASH_REVIEW',created_at=datetime.now(timezone.utc).isoformat(),
        sources=sources,results=results,objective_verified=False,live_activation_supported=False,
        interpretation=['Paper sumR/price and winrate are signal diagnostics, not live account returns.',
            'Selection cash contains only August/September; later zero rows in generic input helper are not evaluated holdouts.',
            'No variant selected: October/November account validation was not performed.',
            'Unknown last-trade exits stay unknown; a known mark path does not verify an executable trade-price fill.'])
    (ROOT/'stock_session_nov19_review.json').write_text(json.dumps(review,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
