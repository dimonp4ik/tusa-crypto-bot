"""Score the frozen stock continuation after its original maturation cutoff."""

import argparse
from datetime import datetime, timezone
import json

from stock_liquid_quote_routes_nov21 import (
    ARCHIVES, HOLD_SECONDS, MATURE_AT, ROOT, capture, freeze, repair_trades, score, score_minutes, write_json,
)


STEM = "stock_quote_continuation_holdout_nov21"
DATA = ROOT / STEM
FIRST_BUCKET = (MATURE_AT - HOLD_SECONDS) // 900 * 900 + 900
MATURE_BY = int(datetime(2026, 11, 21, 21, 30, tzinfo=timezone.utc).timestamp())


def declare():
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "plan.json"
    if path.exists():
        return json.loads(path.read_bytes())
    archives = {"unseen_contract_observer": ARCHIVES["contract_followup"]}
    plan = freeze(archives=archives, first_bucket=FIRST_BUCKET, mature_at=MATURE_BY)
    if not plan["archive_sha256"]["unseen_contract_observer"]:
        raise ValueError("no new mature market observations")
    plan["first_bucket"] = FIRST_BUCKET
    plan["frozen_rules_commit"] = "ebf3b984"
    write_json(path, plan)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("declare", "minute", "capture", "repair", "stress"))
    args = parser.parse_args()
    plan = declare()
    if args.stage == "declare":
        print(json.dumps(plan["counts"], indent=2))
    elif args.stage == "minute":
        score_minutes(data=DATA, stem=STEM)
    elif args.stage == "capture":
        capture(data=DATA)
    elif args.stage == "repair":
        repair_trades(data=DATA)
    else:
        score(stress=True, data=DATA, stem=STEM)


if __name__ == "__main__":
    main()
