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
