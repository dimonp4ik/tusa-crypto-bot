import crypto_xperp_stable_universe_fresh_oct20_audit as audit


def test_after_cutoff_is_strict_and_does_not_recount_boundary():
    rows = [
        {"entry_time": 99}, {"entry_time": 100}, {"entry_time": 101},
    ]
    assert audit.after_cutoff(rows, 100) == [{"entry_time": 101}]


def test_fresh_cache_is_separate_from_frozen_cache():
    assert audit.FRESH_CACHE.resolve() != audit.OLD_CACHE.resolve()
