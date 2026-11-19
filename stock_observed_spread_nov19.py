"""Measure stock spread eligibility from immutable prospective public books."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from src.execution_guard import MAX_ENTRY_SPREAD
from us_cash_session_nov19 import session_bounds


def summarize(rows):
    valid = len(rows)
    spread_ok = sum(spread <= MAX_ENTRY_SPREAD * 10_000 for spread in rows)
    return dict(valid=valid, within_limit=spread_ok,
                median_bps=median(rows) if valid else None,
                within_limit_pct=round(100 * spread_ok / valid, 2) if valid else None)


def study(folder):
    all_books = defaultdict(list)
    candidate_books = defaultdict(list)
    symbols = defaultdict(lambda: defaultdict(list))
    digest_by_file = {}
    invalid = 0
    for path in sorted(folder.glob('bucket_*.json')):
        raw = path.read_bytes()
        window = json.loads(raw)
        if not window['summary']['protocol_ok'] or window['completed_at'] is None:
            continue
        if path.stem != f"bucket_{window['bucket']}":
            raise ValueError('stock archive bucket mismatch')
        digest_by_file[path.name] = hashlib.sha256(raw).hexdigest()
        for observation in window['observations']:
            book = observation['payload'].get('book') or {}
            if not observation['quote_ok'] or not book.get('ok'):
                invalid += 1
                continue
            stamp = float(observation['received_at'])
            bounds = session_bounds(stamp)
            session = 'regular' if bounds and bounds[0] <= stamp < bounds[1] else 'outside_regular'
            spread = float(book['spread_bps'])
            bid, ask = float(book['bid']), float(book['ask'])
            if bid <= 0 or ask <= bid or not math.isclose(
                    spread, 20_000 * (ask-bid) / (ask+bid), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('archived spread does not reconcile')
            for key in ('all', session):
                all_books[key].append(spread)
                symbols[observation['symbol']][key].append(spread)
                if observation['candidate']:
                    candidate_books[key].append(spread)
    if not digest_by_file:
        raise ValueError('no stock book windows')
    return dict(spread_limit_bps=MAX_ENTRY_SPREAD * 10_000, windows=len(digest_by_file),
                invalid_or_missing_books=invalid, sources=digest_by_file,
                books={k: summarize(v) for k,v in all_books.items()},
                candidates={k: summarize(candidate_books[k]) for k in all_books},
                by_symbol={s:{k:summarize(v) for k,v in x.items()} for s,x in sorted(symbols.items())},
                limitation='One prospective public observation per window; no replay of private retries, fills or outcomes.')


if __name__ == '__main__':
    report = study(Path('../tusa stocks/reports/structure_observer_prospective_nov18'))
    target = Path('reports/audit_2026_09_08/stock_observed_spread_nov19.json')
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('windows','invalid_or_missing_books','books','candidates')}))
