import json
import unittest
from types import SimpleNamespace

from app.services.openai_service import OpenAIService


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def service_with(completions, reasoning=False):
    service = OpenAIService.__new__(OpenAIService)
    service.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    service.chat_model = "test-model"
    service.is_reasoning_model = reasoning
    service._retry_with_backoff = lambda func, *args, **kwargs: func(*args, **kwargs)
    return service


class StructuredCompletionTests(unittest.TestCase):
    def test_uses_json_schema_and_parses_response(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ratings":[]}'))]
        )
        completions = FakeCompletions(response=response)
        service = service_with(completions)

        result = service.get_structured_chat_completion(
            system_prompt="System",
            user_message="User",
            schema_name="ratings",
            schema={"type": "object", "properties": {"ratings": {"type": "array"}}},
        )

        self.assertEqual(result, {"ratings": []})
        request = completions.calls[0]
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertEqual(request["temperature"], 0)

    def test_falls_back_to_validated_json_prompt(self):
        completions = FakeCompletions(error=RuntimeError("unsupported"))
        service = service_with(completions)
        fallback_calls = []

        def fallback(**kwargs):
            fallback_calls.append(kwargs)
            return json.dumps({"ratings": [{"id": "a", "score": 0.9}]})

        service.get_chat_completion = fallback
        result = service.get_structured_chat_completion(
            system_prompt="System",
            user_message="User",
            schema_name="ratings",
            schema={"type": "object"},
        )

        self.assertEqual(result["ratings"][0]["id"], "a")
        self.assertEqual(len(fallback_calls), 1)
        self.assertIn("Return ONLY a JSON object", fallback_calls[0]["system_prompt"])


if __name__ == "__main__":
    unittest.main()
