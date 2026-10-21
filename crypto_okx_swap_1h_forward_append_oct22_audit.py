"""Oct-22 append-only replay of the frozen OKX 1H crypto cascade."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import crypto_okx_swap_1h_forward_append_oct16_audit as previous


ROOT = Path("reports/audit_2026_09_08")
PRIOR_GLOBAL = ROOT / "crypto_okx_swap_1h_expanded_oct16" / "manifest.json"
PRIOR_XPERP = ROOT / "xperp_5m_oct16_forward" / "manifest.json"
CURRENT_GLOBAL = ROOT / "crypto_okx_swap_1h_expanded_oct22" / "manifest.json"
CURRENT_XPERP = ROOT / "xperp_5m_oct22_forward" / "manifest.json"
OUT = ROOT / "crypto_okx_swap_1h_forward_append_oct22.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_forward_append_oct22.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    previous.PRIOR_GLOBAL = PRIOR_GLOBAL
    previous.PRIOR_XPERP = PRIOR_XPERP
    previous.CURRENT_GLOBAL = CURRENT_GLOBAL
    previous.CURRENT_XPERP = CURRENT_XPERP
    previous.OUT = OUT
    previous.ROWS_OUT = ROWS_OUT
    previous.main()
    report = json.loads(OUT.read_text(encoding="utf-8"))
    report["protocol"]["append_snapshot"] = "closed public data through 2026-10-22"
    report["sources"][str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "integrity": report["integrity"],
        "append": report["append"],
        "ready_for_live_orders": report["ready_for_live_orders"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
