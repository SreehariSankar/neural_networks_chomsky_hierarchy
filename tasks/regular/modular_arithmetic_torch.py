"""Modular arithmetic without brackets.

Generates samples using modular arithmetic operations, compatible with numpy.
"""

import functools
from typing import Optional, Sequence
import numpy as np

from neural_networks_chomsky_hierarchy.tasks import task_torch as task

# Mapping of operations to integers for encoding
OP_BY_CHARACTER = {'+': 0, '-': 1, '*': 2, '_': 3}


def _replace_subtractions(expression: np.ndarray, modulus: int) -> np.ndarray:
    """Replaces subtractions in an expression by additions with the inverse.

    Args:
        expression: Encoded expression (a 1D array of integers) where subtractions are replaced.
        modulus: The modulus for modular arithmetic.

    Returns:
        Expression with subtractions replaced by additions with the inverse.
    """
    if expression.size < 2:
        return expression

    mask = (expression == modulus + OP_BY_CHARACTER['-'])
    expression = np.where(mask, modulus + OP_BY_CHARACTER['+'], expression)
    expression[2:] *= 1 - 2 * mask[1:-1]
    return expression


def _perform_multiplications(expression: np.ndarray, modulus: int) -> np.ndarray:
    """Performs multiplications in an expression containing only + and *.

    Args:
        expression: Encoded expression with multiplications to evaluate.
        modulus: The modulus for modular arithmetic.

    Returns:
        Array with the results of the multiplications (zero-padded as needed).
    """
    term_ids = np.cumsum(expression == modulus + OP_BY_CHARACTER['+'])[::2]
    max_terms = expression.shape[0] // 2 + 1
    products = np.zeros(max_terms, dtype=int)

    for i in range(len(term_ids)):
        products[term_ids[i]] *= expression[i * 2]

    valid_segment_mask = np.arange(max_terms) <= term_ids[-1]
    return products * valid_segment_mask


def _replace_blanks(expression: np.ndarray, modulus: int) -> np.ndarray:
    """Replaces blank symbols in expression with either `+` or `0`.

    Args:
        expression: Encoded expression with blanks to replace.
        modulus: The modulus for modular arithmetic.

    Returns:
        Array with blanks replaced by either `+` or `0`.
    """
    mask = (expression == OP_BY_CHARACTER['_'] + modulus)
    operator_mask = mask[::2]
    residual_mask = mask[1::2]

    expression[::2] = np.where(operator_mask, OP_BY_CHARACTER['+'] + modulus, expression[::2])
    expression[1::2] = np.where(residual_mask, 0, expression[1::2])
    return expression


def _evaluate_expression(expression: np.ndarray, modulus: int) -> int:
    """Evaluates a modular arithmetic expression."""
    expression = _replace_blanks(expression, modulus)
    expression = _replace_subtractions(expression, modulus)
    additive_terms = _perform_multiplications(expression, modulus)
    return int(np.sum(additive_terms) % modulus)


class ModularArithmetic(task.GeneralizationTask):
    """A task to evaluate simple modular arithmetic expressions."""

    def __init__(self, modulus: int = 5, operators: Optional[Sequence[str]] = None) -> None:
        """Initializes the modular arithmetic task.

        Args:
            modulus: Modulus for arithmetic operations.
            operators: Operators to use in the expressions (default is ['+', '*', '-']).
        """
        self._modulus = modulus
        self._operators = [OP_BY_CHARACTER[op] for op in (operators or ['+', '*', '-'])]

    def sample_batch(self, rng: np.random.Generator, batch_size: int, length: int) -> task.Batch:
        """Returns a batch of modular arithmetic expressions and their labels.

        Args:
            rng: Numpy random generator.
            batch_size: Size of the batch to return.
            length: Length of each sequence; must be odd, so if not, it's adjusted to be odd.

        Returns:
            Dictionary with 'input' as one-hot encoded expressions and 'output' as one-hot encoded labels.
        """
        if length % 2 == 0:
            length -= 1

        remainders = rng.integers(0, self._modulus, size=(batch_size, length // 2 + 1))
        ops = self._modulus + np.array(self._operators)
        operations = rng.choice(ops, (batch_size, length // 2))

        batch = np.empty((batch_size, length), dtype=int)
        batch[:, ::2] = remainders
        expressions = batch.copy()
        expressions[:, 1::2] = operations

        labels = np.array([_evaluate_expression(expr, self._modulus) for expr in expressions])
        one_hot_labels = np.eye(self._modulus)[labels]
        one_hot_expressions = np.eye(self._modulus + len(OP_BY_CHARACTER))[expressions]
        return {'input': one_hot_expressions, 'output': one_hot_labels}

    @property
    def input_size(self) -> int:
        """Returns the input size for the models."""
        return self._modulus + len(OP_BY_CHARACTER)

    @property
    def output_size(self) -> int:
        """Returns the output size for the models."""
        return self._modulus
