import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_cysticcare_responses as generator


class _CompletedProcess:
    def __init__(self, stdout):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


class BenchmarkGeneratorTests(unittest.TestCase):
    def test_request_pins_one_standard_rag_ranking_path(self):
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            response = json.dumps({"status": "success", "response": "ok"})
            return _CompletedProcess(response + generator._HTTP_MARKER + "200")

        with patch.object(generator.subprocess, "run", side_effect=fake_run):
            result = generator.call_chat("How is ADPKD treated?", "test-session")

        payload = json.loads(captured["command"][captured["command"].index("-d") + 1])
        self.assertEqual(result["response"], "ok")
        self.assertFalse(payload["use_adaptive_agent"])
        self.assertFalse(payload["use_cot"])
        self.assertFalse(payload["use_stepback"])
        self.assertTrue(payload["include_retrieval_debug"])

    def test_force_ignores_stale_output_and_regenerates_selected_row(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "output.json"
            input_path.write_text(json.dumps([{"question": "What is ADPKD?"}]), encoding="utf-8")
            output_path.write_text(
                json.dumps([
                    {
                        "question": "What is ADPKD?",
                        "cysticcare_response": "stale answer",
                        "cysticcare_metadata": {},
                    }
                ]),
                encoding="utf-8",
            )

            fresh = {
                "status": "success",
                "response": "fresh answer",
                "sources": [],
                "retrieval_metadata": {"confidence": "high"},
                "retrieved_chunks": [{"id": "paper-a_0"}],
            }
            argv = [
                "generate_cysticcare_responses.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--workers",
                "1",
                "--force",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(generator, "call_chat", return_value=fresh),
            ):
                self.assertEqual(generator.main(), 0)

            output = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(output[0]["cysticcare_response"], "fresh answer")
        self.assertEqual(
            output[0]["cysticcare_metadata"]["retrieved_chunks"], [{"id": "paper-a_0"}]
        )


if __name__ == "__main__":
    unittest.main()
