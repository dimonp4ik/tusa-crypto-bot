import pandas as pd

import crypto_profit_first_official_macro_news_nov1_audit as audit


def test_official_calendar_uses_dst_aware_utc_times():
    events = audit.load_events()
    assert len(events) == 111
    assert events.event_type.value_counts().to_dict() == {
        "CPI": 42, "EMPLOYMENT_SITUATION": 42, "FOMC": 27}
    jan_cpi = events[(events.event_type == "CPI")
                     & (events.event_time_utc == "2022-01-12T13:30:00Z")]
    jul_cpi = events[(events.event_type == "CPI")
                     & (events.event_time_utc == "2022-07-13T12:30:00Z")]
    jan_fomc = events[(events.event_type == "FOMC")
                      & (events.event_time_utc == "2022-01-26T19:00:00Z")]
    jul_fomc = events[(events.event_type == "FOMC")
                      & (events.event_time_utc == "2022-07-27T18:00:00Z")]
    assert len(jan_cpi) == len(jul_cpi) == len(jan_fomc) == len(jul_fomc) == 1


def test_event_mask_has_inclusive_declared_boundaries():
    event = int(pd.Timestamp("2026-10-14T12:30:00Z").timestamp())
    events = pd.DataFrame({
        "event_type": ["CPI"],
        "event_time": [event],
    })
    frame = pd.DataFrame({
        "entry_time": [event - 1801, event - 1800, event,
                       event + 5400, event + 5401],
    })
    assert audit.event_mask(frame, events, ("CPI",), 30, 90).tolist() == [
        False, True, True, True, False]


def test_event_mask_does_not_match_an_unselected_event_type():
    event = int(pd.Timestamp("2026-10-14T12:30:00Z").timestamp())
    events = pd.DataFrame({
        "event_type": ["CPI"],
        "event_time": [event],
    })
    frame = pd.DataFrame({"entry_time": [event]})
    assert not audit.event_mask(
        frame, events, ("FOMC",), 60, 180).iloc[0]
