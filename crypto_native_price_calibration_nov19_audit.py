"""One daily correction hypothesis; July/August threshold, later shared cash."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from crypto_native_price_features_nov19 import FEATURES
from crypto_native_price_pool_nov19 import load_pool,ROOT,MANIFEST,META
from crypto_native_price_nov19_audit import (TRAIN_END,SELECT_END,THRESHOLDS,
    STOCK_PATH,SPEC_PATH,select_threshold,gated,later_report)
from crypto_native_long_phase_nov19 import attach_long_phase,PHASES
from crypto_native_price_calibration_nov19 import (calibrated_scores,HISTORY_DAYS,
    SHRINK_TIMESTAMPS,CORRECTION_WEIGHT)
from shared_factor_account_nov14_audit import stock_ready_rows


def main():
    model_path=ROOT/'crypto_native_price_nov19_model.pkl'
    source_paths=(Path(__file__),Path('crypto_native_price_calibration_nov19.py'),
        Path('crypto_native_long_phase_nov19.py'),Path('crypto_native_price_nov19_audit.py'),
        ROOT/'crypto_native_price_nov19.json',model_path,META,MANIFEST,STOCK_PATH,SPEC_PATH)
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=sources,
        history_days=HISTORY_DAYS,shrink_timestamps=SHRINK_TIMESTAMPS,correction_weight=CORRECTION_WEIGHT,
        calibration_start=TRAIN_END,thresholds=THRESHOLDS,selection_end=SELECT_END,
        minimum_distinct_history_times=dict(phase=50,family=100,global_fallback=200),
        no_training_period_labels_for_calibration=True,daily_midnight_full_horizon_purge=True,
        hypothesis_origin='daily residual correction after observing base model phase-dependent later errors')
    declaration=ROOT/'crypto_native_price_calibration_nov19_protocol.json'
    if declaration.exists():
        prior=json.loads(declaration.read_text(encoding='utf-8'))
        if prior['sources']!=sources:raise ValueError('calibration_declared_source_changed')
        protocol=prior
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    frame,meta=load_pool();candles={}
    for symbol,item in json.loads(MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        if symbol not in ('BTCUSDT','ETHUSDT'):continue
        p=Path(item['path']);raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('calibration_phase_source_changed')
        candles[symbol]=pickle.loads(raw)
    # Only the already-fitted May/June model is used. No later refits.
    estimator=pickle.loads(model_path.read_bytes())
    if tuple(estimator['features'])!=FEATURES or estimator['protocol']['train_end']!=TRAIN_END:
        raise ValueError('unexpected_frozen_native_model')
    frame=frame[frame.entry_time.ge(TRAIN_END)].copy()
    frame=attach_long_phase(frame,candles)
    phase_columns=['feature__BTC_phase_'+phase for phase in PHASES]
    frame['long_btc_regime']=np.asarray(PHASES)[frame[phase_columns].to_numpy().argmax(axis=1)]
    with threadpool_limits(limits=2):
        frame['raw_prediction']=estimator['model'].predict(frame[['feature__'+n for n in FEATURES]].to_numpy())
    # Generate and lock selection scores before processing later corrections.
    selection,selection_decisions=calibrated_scores(frame[frame.entry_time.lt(SELECT_END)],calibration_start=TRAIN_END)
    stock=stock_ready_rows(pd.read_pickle(STOCK_PATH)).assign(bot='stocks',sequential_banked=False,near_exit_time=np.nan)
    stock['sequential_runner_r']=stock.net_r;specs=json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    chosen,selection_report=select_threshold(selection,stock,specs)
    print('calibration_selection',chosen,selection_report,flush=True)
    base=json.loads((ROOT/'crypto_native_price_nov19.json').read_text(encoding='utf-8'))
    base_score=base['selection'][str(base['chosen_threshold'])]['score']
    candidate_score=selection_report[str(chosen)]['score'] if chosen is not None else None
    improves=chosen is not None and candidate_score>base_score+1e-12
    later_result=None;decisions=selection_decisions
    if improves:
        corrected,decisions=calibrated_scores(frame,calibration_start=TRAIN_END)
        # Later rows cannot change any selection score, including its history.
        np.testing.assert_array_equal(selection.predicted_price_return.to_numpy(),
            corrected.loc[selection.index,'predicted_price_return'].to_numpy())
        later=corrected[corrected.entry_time.ge(SELECT_END)]
        accepted=gated(later,chosen);path=ROOT/'crypto_native_price_calibration_nov19_later.pkl'
        accepted.to_pickle(path);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        later_result=later_report(accepted,stock,specs)
        print('calibration_later',{m:dict(crypto=v['return_contribution_pct']['crypto'],
            crypto_closed=v['crypto_closed'],crypto_wr=v['crypto_winrate_pct'])
            for m,v in later_result['crypto']['calendar_months'].items()},flush=True)
    report=dict(status='PREQUENTIAL_PRICE_CALIBRATION_RESEARCH_ONLY',objective_verified=False,
        live_activation_supported=False,protocol=protocol,selection=selection_report,chosen_threshold=chosen,
        base_selection_score=base_score,calibrated_selection_score=candidate_score,improves_selection=improves,
        daily_calibration=decisions,later=later_result,sources=sources,limitations=base['limitations']+[
            'Correction chosen after inspecting base-model later errors; no pristine holdout is claimed.',
            'Every midnight uses only prior 30-day raw candidates with full horizon and exit strictly before midnight.',
            'Calibration excludes May/June fit labels, uses timestamp weights and minimum distinct times, not independence claims.',
            'Hypothetical outcomes remain available when no private trade opened; account risk rejection is not a model loss label.',
            'Cash threshold chosen only from July/August; later outcomes may update the fixed daily rule, never threshold.'])
    (ROOT/'crypto_native_price_calibration_nov19.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
