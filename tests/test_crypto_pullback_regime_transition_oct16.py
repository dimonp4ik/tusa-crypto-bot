import unittest

import pandas as pd

import crypto_pullback_regime_transition_oct16_audit as audit


class PullbackRegimeTransitionTests(unittest.TestCase):
    def test_directional_transition_features_are_symmetric(self):
        frame = pd.DataFrame({
            "direction": [1, -1],
            "family_pullback": [1.0, 1.0],
            "ret24": [-.01, .01],
            "ret72": [.03, -.03],
            "ret168": [.04, -.04],
            "btc_ret24": [-.02, .02],
            "cross_z24": [-1.0, 1.0],
            "breadth24": [.25, .75],
        })
        enriched = audit.enrich(frame)
        risks = dict(audit.candidate_risks(enriched))
        self.assertEqual(enriched.aligned_ret24.tolist(), [-.01, -.01])
        self.assertEqual(enriched.breadth_alignment.tolist(), [.25, .25])
        self.assertTrue(risks["symbol_btc_breadth_oppose"].all())


if __name__ == "__main__":
    unittest.main()
