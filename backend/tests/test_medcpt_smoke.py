"""Opt-in real-model smoke coverage for the local MedCPT reranker.

Run explicitly after model dependencies/cache are available:
``RUN_MEDCPT_INTEGRATION=1 python -m unittest tests.test_medcpt_smoke``.
The normal suite remains hermetic and uses deterministic scorer fakes instead.
"""

import math
import os
import unittest

from app.services.cross_encoder_reranker import CrossEncoderConfig, MedCPTCrossEncoder


@unittest.skipUnless(
    os.getenv("RUN_MEDCPT_INTEGRATION") == "1",
    "Set RUN_MEDCPT_INTEGRATION=1 to download/load the real MedCPT model",
)
class MedCPTSmokeTests(unittest.TestCase):
    def test_real_model_scores_medical_query_passage_pairs(self):
        scorer = MedCPTCrossEncoder(
            CrossEncoderConfig(
                model_name="ncbi/MedCPT-Cross-Encoder",
                device="cpu",
                local_files_only=False,
                batch_size=2,
                max_length=512,
            )
        )
        scores = scorer.score_pairs(
            "Does tolvaptan slow kidney-function decline in ADPKD?",
            [
                "In the TEMPO 3:4 trial, tolvaptan reduced the rate of decline "
                "in estimated glomerular filtration rate among adults with ADPKD.",
                "The paper describes the history of renal ultrasound imaging "
                "techniques in paediatric care.",
            ],
        )

        self.assertEqual(len(scores), 2)
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
