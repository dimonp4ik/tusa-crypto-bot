from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_tick_path_nov19 import Trade, resolve_ticks


def test_delayed_protection_applies_only_after_its_observation_boundary():
    signal = dict(epoch=0, bot="crypto", direction="LONG", entry=100., stop=99., target=102.)
    tape = [Trade(stamp, i, price, 1.) for i, (stamp, price) in enumerate(
        ((1_000, 100.5), (61_000, 99.85), (62_000, 99.49), (63_000, 101.), (121_000, 99.95)), 1)]
    times = [r.ms for r in tape]
    params = dict(signal=signal, entered_at=500, entry_px=100., tick_size=.01, amend_seconds=60)
    immediate = resolve_ticks(tape, times, **params)
    delayed = resolve_ticks(tape, times, runner_policy="be_1r", **params)
    half = resolve_ticks(tape, times, runner_policy="half_stop_then_1r", **params)
    assert (immediate["code"], immediate["end_ms"]) == ("BE", 61_000)
    assert (delayed["code"], delayed["end_ms"]) == ("BE", 121_000)
    assert (half["code"], half["end_ms"]) == ("TP1_HALF_STOP", 62_000)
    assert immediate["bank"] == delayed["bank"] == half["bank"] == {"ms": 1_000, "price": 100.5}


def test_cash_chooses_delay_only_when_the_clients_requested_partial_cannot_be_placed():
    baseline = dict(known=True, code="BE", end_ms=61_000, bank={"ms": 1_000}, be_level=100.)
    deferred = dict(baseline, end_ms=121_000, code="TARGET")
    proposal = dict(symbol="TEST", bot="crypto", direction="LONG", entry_ms=500,
                    entry=100., stop=99., target=102., entry_status="eligible",
                    path={60: baseline}, runner_paths={"be_1r": {60: deferred}})
    spec = {"ctVal": .1, "lotSz": 1., "minSz": 1., "ctValCcy": "TEST"}
    prices = {500: 99.98, 1_200: 100.49, 61_200: 100., 121_200: 101.99}
    books = {"TEST": {
        stamp: {"ts": stamp-1, "bids": [(bid, 99.)], "asks": [(bid+.02, 99.)]}
        for stamp, bid in prices.items()
    }}

    def cash(setting, policy=None):
        return simulate([proposal], {"TEST": spec}, {"TEST": .01}, books,
                        clock=60, lag=200, setting=setting, unbankable_runner=policy)

    no_partial = cash("requested5")
    delayed = cash("requested5", "be_1r")
    assert no_partial["closed"][0]["bank_qty"] == delayed["closed"][0]["bank_qty"] == 0
    assert delayed["closed"][0]["exit_ms"] > no_partial["closed"][0]["exit_ms"]
    banked = cash("half_minimum")
    assert banked["closed"][0]["bank_qty"] == 1.
    assert banked == cash("half_minimum", "be_1r")
