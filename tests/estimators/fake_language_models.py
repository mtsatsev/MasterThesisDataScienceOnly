"""Fake language models for the estimator tests: fixed numbers, no model library."""


class FixedNextTokenModel:
    """Return the same next-token log-probabilities for every prompt, and remember
    every call."""

    def __init__(self, logprobs: dict[str, float]):
        self.logprobs = logprobs
        self.calls = []

    def next_token_logprobs(
        self, prompts: list[str], tokens: list[str]
    ) -> list[dict[str, float]]:
        self.calls.append((prompts, tokens))
        return [{token: self.logprobs[token] for token in tokens} for _ in prompts]


class StoredContinuationModel:
    """Return stored per-token log-probabilities per (prefix, continuation), [-1.0]
    for unknown requests, and remember every call."""

    def __init__(
        self, token_logprobs: dict[tuple[str, str], list[float]] | None = None
    ):
        self.token_logprobs = token_logprobs or {}
        self.calls = []

    def continuation_token_logprobs(
        self, requests: list[tuple[str, str]]
    ) -> list[list[float]]:
        self.calls.append(requests)
        return [self.token_logprobs.get(request, [-1.0]) for request in requests]
