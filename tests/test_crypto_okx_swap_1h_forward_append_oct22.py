from pathlib import Path

import crypto_okx_swap_1h_forward_append_oct22_audit as audit


def test_oct22_wrapper_uses_oct16_as_prior_snapshot():
    assert audit.PRIOR_GLOBAL == Path(
        "reports/audit_2026_09_08/crypto_okx_swap_1h_expanded_oct16/manifest.json")
    assert audit.PRIOR_XPERP == Path(
        "reports/audit_2026_09_08/xperp_5m_oct16_forward/manifest.json")
    assert audit.CURRENT_GLOBAL == Path(
        "reports/audit_2026_09_08/crypto_okx_swap_1h_expanded_oct22/manifest.json")
    assert audit.CURRENT_XPERP == Path(
        "reports/audit_2026_09_08/xperp_5m_oct22_forward/manifest.json")
