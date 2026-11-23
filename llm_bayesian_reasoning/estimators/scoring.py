import math


def sigmoid(logit: float) -> float:
    """
    Return 1 / (1 + e^-logit) without overflowing for large logits.

    Args:
        logit (float):
            The input logit.

    Returns:
        float: The sigmoid of the logit.
    """
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    return math.exp(logit) / (1 + math.exp(logit))


def softmax(values: list[float]) -> list[float]:
    """
    Return the softmax of the values, shifted by their maximum to stay finite.

    Args:
        values (list[float]):
            The log-scores to normalise.

    Returns:
        list[float]: The probabilities, in the same order, summing to 1.
    """
    largest = max(values)
    exponentials = [math.exp(value - largest) for value in values]
    total = sum(exponentials)
    return [exponential / total for exponential in exponentials]


def check_temperature(temperature: float) -> None:
    """
    Reject a temperature that is not positive and finite.

    Args:
        temperature (float):
            The temperature to check.
    """
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
