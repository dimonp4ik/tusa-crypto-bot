import unittest

import live_stop_cluster_oct16_audit as audit


def row(identifier, opened, direction="LONG", entry=100, exit_price=101, realized_r=1):
    return {
        "id": str(identifier), "opened_at": str(opened), "direction": direction,
        "entry_price": str(entry), "exit_price": str(exit_price),
        "realized_r": str(realized_r), "status": "SL_HIT" if realized_r < 0 else "TP1_TRAIL",
    }


class LiveStopClusterAuditTests(unittest.TestCase):
    def test_short_return_is_signed_and_account_uses_client_fraction(self):
        rows = audit.prepare([row(1, 1, "SHORT", 100, 99, 1)])
        self.assertAlmostEqual(rows[0]["_return"], .01)
        path = audit.account_path(rows, margin_fraction=.03, leverage=10)
        self.assertAlmostEqual(path["return_pct"], .3)

    def test_rolling_cap_is_causal_and_uses_accepted_entries_only(self):
        rows = audit.prepare([
            row(1, 0), row(2, 10), row(3, 20), row(4, 3605), row(5, 3610),
        ])
        selected = audit.causal_cap(rows, cap=2, seconds=3600)
        self.assertEqual([r["id"] for r in selected], ["1", "2", "4", "5"])

    def test_direction_cap_does_not_mix_long_and_short(self):
        rows = audit.prepare([
            row(1, 0, "LONG"), row(2, 10, "SHORT", 100, 99, -1),
            row(3, 20, "LONG"), row(4, 30, "SHORT", 100, 99, -1),
        ])
        selected = audit.causal_cap(rows, cap=1, same_direction=True)
        self.assertEqual([r["id"] for r in selected], ["1", "2"])

    def test_cluster_membership_is_diagnostic_and_symmetric(self):
        rows = audit.prepare([row(1, 0), row(2, 899), row(3, 1799)])
        clustered, isolated = audit.cluster_membership(rows, seconds=900)
        self.assertEqual([r["id"] for r in clustered], ["1", "2", "3"])
        self.assertEqual(isolated, [])

    def test_structure_relation(self):
        self.assertEqual(audit.relation({"direction": "LONG"}, {"trend": "bull"}), "aligned")
        self.assertEqual(audit.relation({"direction": "SHORT"}, {"trend": "bull"}), "opposed")
        self.assertEqual(audit.relation({"direction": "LONG"}, {"trend": "range"}), "range_or_missing")


if __name__ == "__main__":
    unittest.main()
