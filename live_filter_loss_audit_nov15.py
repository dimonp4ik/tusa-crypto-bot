"""Reconstruct pre-entry stock features from hashed public data, not outcomes."""
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('reports/audit_2026_09_08/live_filter_review_nov15')
STOCK = Path(__file__).resolve().parent.parent/'tusa stocks'


def load_snapshot(path):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    candles = {}
    for symbol, meta in manifest['symbols'].items():
        if not meta.get('path'):
            continue
        source = Path(meta['path'])
        if not source.is_absolute() and not source.exists():
            source = STOCK / source
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == meta['sha256'], symbol
        value = pickle.loads(raw)
        assert all(b-a == 900 for a,b in zip(value['time'],value['time'][1:])), symbol
        candles[symbol] = value
    return candles


def main():
    spec = importlib.util.spec_from_file_location('stock_native', STOCK/'src/stock_xperp_native_shadow.py')
    native = importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    old_path = STOCK/'reports/audit_2026_09_08/stock_xperp_forward_nov14_merged/manifest.json'
    new_path = ROOT/'stock_15m/manifest.json'
    old, new = load_snapshot(old_path), load_snapshot(new_path)
    from append_candle_snapshot import merge
    combined = {s: merge(old[s], new[s])[0] for s in new if s in old}
    exported = json.loads((ROOT/'stock_signals.json').read_text())
    setups = {s['signal_id']:s for s in exported['setups']}
    output = []
    for signal in reversed(exported['signals']):
        setup = setups.get(signal['id'])
        if not setup or setup['source'] != 'stock_xperp_native_shadow':
            continue
        stamp = int(setup['signal_bar_ts'])
        cut = {}
        for symbol, data in combined.items():
            indices = [i for i,t in enumerate(data['time']) if t <= stamp][-200:]
            if indices:
                cut[symbol] = {k:[v[i] for i in indices] for k,v in data.items()}
        diag = {}
        candidates = native.native_shadow_setups(cut, now=stamp+914, diagnostics=diag)
        candidate = next((r for r in candidates if r['symbol']==signal['symbol']),None)
        data = cut[signal['symbol']]
        last = len(data['time'])-1
        o,h,l,c = (float(data[k][last]) for k in ('open','high','low','close'))
        row = dict(signal_id=signal['id'],symbol=signal['symbol'],direction=signal['direction'],
            opened_utc=datetime.fromtimestamp(signal['opened_at'],timezone.utc).isoformat(),
            entry=signal['entry_price'],sl=signal['sl'],target=signal['tp1'],
            status=signal['status'],r=signal['realized_r'],
            reconstructed_candidate=candidate,reconstruction_result=diag['result'],
            signal_bar=dict(open=o,high=h,low=l,close=c,volume=data['volume'][last]),
            close_location=(c-l)/(h-l) if h>l else None,
            signal_bar_return=c/o-1,
            return_4=c/data['close'][-5]-1,
            return_16=c/data['close'][-17]-1,
            entry_drift_pct=100*(signal['entry_price']/c-1),
            publication_delay_s=signal['opened_at']-(stamp+900),
            book_kind=setup['book_sample_kind'],book_spread_bps=setup['book_spread_bps'],
            book_observed_after_entry_s=(setup['book_observed_ts']-signal['opened_at']
                if setup['book_observed_ts'] else None))
        output.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='reconstructed_candidate'}),flush=True)
    report=dict(status='DIAGNOSTIC_NOT_FILTER_VALIDATION',rows=output,
        limitations=['Public candle reconstruction, not the exact live fetch snapshot.',
            'Current rule version; older trades may have used different guards.',
            'Post-publication order book is not a pre-entry feature.',
            'No actual client fills/balances are exported. No monthly profit claim.'],
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            old_path,new_path,Path(__file__),STOCK/'src/stock_xperp_native_shadow.py',
            ROOT/'stock_signals.json')})
    (ROOT/'stock_reconstruction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
