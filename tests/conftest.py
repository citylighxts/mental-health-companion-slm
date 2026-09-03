"""Shared test helpers."""
from types import SimpleNamespace


class FakeMessages:
    """Stand-in for client.messages — pops canned text responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeMessages ran out of canned responses")
        text = self._responses.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)
