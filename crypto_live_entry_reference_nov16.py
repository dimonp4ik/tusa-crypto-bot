"""Compare observed model inputs with the later finalized native bar open."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from public_candle_window import fetch_window
from src.crypto_profit_first_material_shadow_router import model_probability

ROOT=Path('reports/audit_2026_09_08/live_filter_review_nov15')


def main():
    paths=[ROOT/'crypto_signals.json',ROOT/'crypto_latest_after_rollback_nov16.json']
    signals={};setups={}
    for path in paths:
        export=json.loads(path.read_text())
        signals.update({r['id']:r for r in export['signals']})
        setups.update({r['signal_id']:r for r in export['setups']})
    manifest=ROOT/'crypto_15m/manifest.json';meta=json.loads(manifest.read_text())
    folder=ROOT/'crypto_first_trade_windows';folder.mkdir(exist_ok=True)
    def inspect(row):
        start=int(row['strategy_entry_bar_ts']);end=start+900
        inst=meta['symbols'][row['symbol']]['instrument']
        coarse,ch=fetch_window(inst,start,end,900,folder)
        minutes,mh=fetch_window(inst,start,end,60,folder)
        first_minute=next((t for i,t in enumerate(minutes['time']) if minutes['volume'][i]>0),None)
        first=None;price=None;sh=None
        if first_minute is not None:
            seconds,sh=fetch_window(inst,first_minute,first_minute+60,1,folder)
            j=next(i for i,v in enumerate(seconds['volume']) if v>0)
            first=seconds['time'][j];price=seconds['open'][j]
            if abs(price-coarse['open'][0])>1e-8*price:
                raise ValueError('first traded second disagrees with native 15m open')
        setup=setups[row['id']];context=setup.get('decision_context_json')
        context=json.loads(context)['decision_evidence'] if context else None
        saved_score=recalculated=None;reference=None
        if context:
            reference=context['entry_reference_price'];saved_score=context['model_quality']
            recalculated=model_probability(symbol=row['symbol'],signal=context,context=context,
                entry_time=start,risk_pct=context['model_risk_pct'])
            if abs(saved_score-recalculated)>1e-12:
                raise ValueError('stored deterministic model score does not reproduce')
        return dict(id=row['id'],symbol=row['symbol'],direction=row['direction'],
            opened_at=row['opened_at'],entry=row['entry_price'],r=row['realized_r'],
            finalized_15m_open=coarse['open'][0],first_traded_second=first,
            first_trade_delay_s=first-start if first is not None else None,
            first_trade_after_publication=first>row['opened_at'] if first is not None else None,
            observed_model_reference=reference,stored_score=saved_score,recalculated_score=recalculated,
            reference_differs_from_final_open=(abs(reference/coarse['open'][0]-1)>1e-10 if reference else None),
            source_hashes=dict(coarse=ch,minutes=mh,first_trade_seconds=sh))
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for r in pool.map(inspect,sorted(signals.values(),key=lambda r:r['opened_at'])):
            rows.append(r);print(json.dumps(r),flush=True)
    summary=dict(signals=len(rows),first_trade_after_publication=sum(r['first_trade_after_publication'] is True for r in rows),
        stored_scores_reproduced=sum(r['stored_score'] is not None for r in rows),
        stored_reference_differs_from_final_open=sum(r['reference_differs_from_final_open'] is True for r in rows))
    report=dict(status='OBSERVED_ENTRY_REFERENCE_AUDIT_NOT_ACCOUNT_PNL',summary=summary,rows=rows,
        limitations=['Public last-trade prices do not establish an executable bid/ask market fill.',
            'Only saved decision snapshots can verify the actual observed model reference.',
            'Earlier rows without snapshots remain explicitly unknown.',
            'Less-than-one-second ordering of the first trade and publication remains unresolved.'],
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,manifest,
            Path(__file__),Path('src/frozen_crypto_profit_first_material_2r.json'),
            Path('src/crypto_profit_first_material_shadow_router.py')]})
    (ROOT/'crypto_live_entry_reference_nov16.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
