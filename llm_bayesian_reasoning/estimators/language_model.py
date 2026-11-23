from typing import Protocol


class LanguageModel(Protocol):
    """The model operations the estimators need, with no model library in sight."""

    def next_token_logprobs(
        self, prompts: list[str], tokens: list[str]
    ) -> list[dict[str, float]]:
        """
        Return, per prompt, the log-probability of each token coming next.
        Args:
            prompts (list[str]):
                The list of prompt strings to evaluate.
            tokens (list[str]):
                The list of tokens for which to compute log-probabilities.

        Returns:
            list[dict[str, float]]: A list of dictionaries, one per prompt, mapping
                each token to its log-probability.
        """
        ...

    def continuation_token_logprobs(
        self, requests: list[tuple[str, str]]
    ) -> list[list[float]]:
        """
        Return, per (prefix, continuation) request, the log-probability of each
        continuation token after the prefix.

        Args:
            requests (list[tuple[str, str]]):
                The (prefix, continuation) pairs to evaluate.

        Returns:
            list[list[float]]: One list per request, one log-probability per
                continuation token.
        """
        ...

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Return the text cut to its first max_tokens tokens.

        Args:
            text (str):
                The text to cut.
            max_tokens (int):
                The number of tokens to keep.

        Returns:
            str: The decoded first max_tokens tokens of the text.
        """
        ...
