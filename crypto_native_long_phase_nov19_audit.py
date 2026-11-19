"""A missing long-phase hypothesis selected on July-August shared cash only."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from crypto_native_price_features_nov19 import FEATURES,mature_before
from crypto_native_price_pool_nov19 import load_pool,ROOT,MANIFEST,META
from crypto_native_price_nov19_audit import (MODEL,TRAIN_END,SELECT_END,THRESHOLDS,
    STOCK_PATH,SPEC_PATH,select_threshold,gated,later_report)
from crypto_native_long_phase_nov19 import LONG_FEATURES,attach_long_phase
from shared_factor_account_nov14_audit import stock_ready_rows


def main():
    source_paths=(Path(__file__),Path('crypto_native_long_phase_nov19.py'),
        Path('crypto_native_price_nov19_audit.py'),Path('src/crypto_venue_router.py'),
        ROOT/'crypto_native_price_nov19.json',MANIFEST,META,STOCK_PATH,SPEC_PATH)
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=sources,model=MODEL,
        extra_features=LONG_FEATURES,thresholds=THRESHOLDS,train_end=TRAIN_END,selection_end=SELECT_END,
        hypothesis_origin='added after observing long-phase concentration in base-model October losses',
        compare_to_base_on_selection_only=True,later_evaluated_only_if_selection_improves=True)
    declaration=ROOT/'crypto_native_long_phase_nov19_protocol.json'
    if declaration.exists():
        old=json.loads(declaration.read_text(encoding='utf-8'))
        if old['sources']!=sources:raise ValueError('long_phase_declared_source_changed')
        protocol=old
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    frame,meta=load_pool();candles={}
    for symbol,item in json.loads(MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        if symbol not in ('BTCUSDT','ETHUSDT'):continue
        p=Path(item['path']);raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('long_context_source_changed')
        candles[symbol]=pickle.loads(raw);sources[str(p)]=item['sha256']
    frame=attach_long_phase(frame,candles)
    columns=['feature__'+n for n in (*FEATURES,*LONG_FEATURES)]
    fit=frame[mature_before(frame,TRAIN_END)]
    if not np.isfinite(fit[columns+['runner_price_return']].to_numpy()).all():raise ValueError('invalid_fit')
    weights=1./fit.groupby('entry_time').entry_time.transform('size').to_numpy();weights*=len(weights)/weights.sum()
    model=HistGradientBoostingRegressor(**MODEL)
    stock=stock_ready_rows(pd.read_pickle(STOCK_PATH)).assign(bot='stocks',sequential_banked=False,near_exit_time=np.nan)
    stock['sequential_runner_r']=stock.net_r;specs=json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    base=json.loads((ROOT/'crypto_native_price_nov19.json').read_text(encoding='utf-8'))
    base_score=base['selection'][str(base['chosen_threshold'])]['score'] if base['chosen_threshold'] is not None else -float('inf')
    with threadpool_limits(limits=2):
        model.fit(fit[columns].to_numpy(),fit.runner_price_return.to_numpy(),sample_weight=weights)
        selection=frame[frame.entry_time.ge(TRAIN_END)&frame.entry_time.lt(SELECT_END)].copy()
        selection['predicted_price_return']=model.predict(selection[columns].to_numpy())
        chosen,selection_report=select_threshold(selection,stock,specs)
        candidate_score=selection_report[str(chosen)]['score'] if chosen is not None else -float('inf')
        improves=chosen is not None and candidate_score>base_score+1e-12
        print('long_phase_selection',dict(threshold=chosen,score=candidate_score,base_score=base_score,improves=improves),flush=True)
        later_result=None
        if improves:
            later=frame[frame.entry_time.ge(SELECT_END)].copy()
            later['predicted_price_return']=model.predict(later[columns].to_numpy())
            accepted=gated(later,chosen)
            path=ROOT/'crypto_native_long_phase_nov19_later.pkl';accepted.to_pickle(path)
            sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            later_result=later_report(accepted,stock,specs)
            print('long_phase_later',later_result,flush=True)
    model_path=ROOT/'crypto_native_long_phase_nov19_model.pkl'
    model_path.write_bytes(pickle.dumps(dict(model=model,features=(*FEATURES,*LONG_FEATURES),protocol=protocol),protocol=5))
    sources[str(model_path)]=hashlib.sha256(model_path.read_bytes()).hexdigest()
    report=dict(status='LONG_PHASE_ABLATION_RESEARCH_ONLY',objective_verified=False,live_activation_supported=False,
        protocol=protocol,training_rows=len(fit),pool_rows=len(frame),chosen_threshold=chosen,
        selection=selection_report,base_selection_score=base_score,improves_selection=improves,
        later=later_result,sources=sources,limitations=base['limitations']+[
            'This extra feature hypothesis was suggested by later-period base-model errors; no independent holdout is claimed.',
            'Model/threshold promotion is decided only from July/August cash; poor later results never alter that threshold.',
            'Long phases describe closed price history, not causal proof of why an individual stop happened.'])
    (ROOT/'crypto_native_long_phase_nov19.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
