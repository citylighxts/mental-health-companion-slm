"""Shared test helpers."""
from types import SimpleNamespace


class FakeMessages:
    """Stand-in for client.messages — pops canned responses in order.

    A canned response that is an Exception instance is raised instead of returned,
    so tests can simulate transient API failures.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeMessages ran out of canned responses")
        text = self._responses.pop(0)
        if isinstance(text, BaseException):
            raise text
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)
