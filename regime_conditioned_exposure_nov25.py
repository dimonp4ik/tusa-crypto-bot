"""Compare BTC exposure limits that are conditional on observable market state."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

import combined_breakout_exposure_nov25 as combined
import shared_correlated_exposure_nov25 as exposure
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


BASE_ADMISSION = exposure.Admission
POLICIES = ('opposed_four_hours', 'opposed_or_high_volatility')
CURRENT_WINDOW = 16
VOLATILITY_MULTIPLIER = 1.25


def market_state(returns, loading):
    past = returns[:-CURRENT_WINDOW]
    recent = returns[-CURRENT_WINDOW:]
    past = past[np.isfinite(past)]
    recent = recent[np.isfinite(recent)]
    if len(past)<48 or len(recent)<12:
        return None
    opposing = loading*float(recent.sum())<0
    volatile = float(np.mean(recent**2)) > VOLATILITY_MULTIPLIER**2 * float(np.mean(past**2))
    return opposing, volatile


class ConditionalAdmission(BASE_ADMISSION):
    def __init__(self,pairs,policy,*,exact,condition):
        super().__init__(pairs,policy,exact=exact)
        self.condition = condition

    def assess(self,symbol,bot,side,epoch,held):
        hits = super().assess(symbol,bot,side,epoch,held)
        if not hits or self.policy != combined.EXPOSURE:
            return hits
        direction = exposure.signed(side)*self.pairs.get(symbol,exposure.REF,epoch)
        state = market_state(self.pairs.window(exposure.REF,epoch),direction)
        if state is None:
            self.counts['market_state_unknown'] += 1
            return hits
        opposing,volatile = state
        keep_cap = opposing if self.condition=='opposed_four_hours' else opposing or volatile
        self.counts[f'state_opposed_{opposing}_high_volatility_{volatile}'] += 1
        return hits if keep_cap else []


def with_condition(policy, operate, *args):
    def factory(pairs,name,*,exact):
        return ConditionalAdmission(pairs,name,exact=exact,condition=policy)

    with patch.object(exposure,'Admission',factory):
        return operate(*args)


def exact_cases(mode):
    operation = {
        'older':combined.older, 'alternate20':combined.alternate20,'alternate21':combined.alternate21,
        'fresh':lambda:combined.captured('fresh'),'native_joint':lambda:combined.captured('native_joint'),
    }[mode]
    original = combined.exact

    def score(*args):
        return {'conditions':{n:with_condition(n,original,*args) for n in POLICIES}}

    with patch.object(combined,'exact',score):
        return operation()


def monthly_cases():
    return {n:with_condition(n,combined.monthly) for n in POLICIES}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('monthly','older','alternate20','alternate21','fresh','native_joint'))
    mode = parser.parse_args().mode
    source = ROOT/f'combined_breakout_exposure_{mode}_nov25.json'
    control = read_json(source)
    if control['source_sha256'][Path(combined.__file__).name] != file_hash(Path(combined.__file__)):
        raise ValueError('combined control runner changed')
    output = monthly_cases() if mode=='monthly' else exact_cases(mode)
    sources = (Path(__file__),Path(combined.__file__),Path(exposure.__file__),source)
    write_json(ROOT/f'regime_conditioned_exposure_{mode}_nov25.json',dict(
        cases=output,policies=POLICIES,volatility_threshold=VOLATILITY_MULTIPLIER,
        source_sha256={p.name:file_hash(p) for p in sources},live_activation_supported=False,limitations=[
            'Condition variants are compared against the frozen unconditional combined replay.',
            'Volatility is four-hour RMS against the earlier 20 hours. Returns must come from active closed bars.',
            'The rule stays effective when insufficient price evidence prevents classification.',
            'The base and weak-only scenarios are unchanged. Exposure-only and combined use the tested conditional rule.',
            'Public-book and candle replays do not represent private execution or guaranteed return.',
        ]))


if __name__=='__main__':
    main()
