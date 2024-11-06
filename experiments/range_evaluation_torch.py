"""Evaluation of a network on sequences of different lengths."""

import dataclasses
import random
from typing import Any, Callable, Mapping
import numpy as np
from tqdm import tqdm

# Type alias for batch data
Batch = Mapping[str, np.ndarray]


@dataclasses.dataclass
class EvaluationParams:
    """Parameters used for range evaluation of networks."""
    model: Callable  # Function to apply the model.
    params: Any  # Model parameters.

    accuracy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    sample_batch: Callable[[np.random.Generator, int, int], Batch]

    max_test_length: int
    total_batch_size: int
    sub_batch_size: int  # Used to avoid memory overflow.

    is_autoregressive: bool = False


def range_evaluation(
        eval_params: EvaluationParams,
        use_tqdm: bool = False,
) -> list[Mapping[str, Any]]:
    """Evaluates the model on sequences of increasing lengths and logs the results.

    Args:
        eval_params: The evaluation parameters, see above.
        use_tqdm: Whether to use a progress bar with tqdm.

    Returns:
        A list of dictionaries containing accuracy metrics for each length.
    """
    model = eval_params.model
    params = eval_params.params

    # Set random seeds for reproducibility
    random.seed(1)
    np.random.seed(1)
    rng = np.random.default_rng(1)

    # Prepare results and iterate through different sequence lengths
    results = []
    lengths = range(1, eval_params.max_test_length + 1)
    if use_tqdm:
        lengths = tqdm(lengths)

    for length in lengths:
        sub_accuracies = []
        for _ in range(eval_params.total_batch_size // eval_params.sub_batch_size):
            # Sample batch data for evaluation
            batch = eval_params.sample_batch(rng, eval_params.sub_batch_size, length)

            # Apply model function
            if eval_params.is_autoregressive:
                outputs = model(params, batch['input'], batch['output'], sample=True)
            else:
                outputs = model(params, batch['input'])

            # Calculate accuracy for the batch
            sub_accuracies.append(float(np.mean(eval_params.accuracy_fn(outputs, batch['output']))))

        # Log and store accuracy for the current length
        log_data = {
            'length': length,
            'accuracy': np.mean(sub_accuracies),
        }
        print(log_data)  # Using print as a substitute for logging
        results.append(log_data)

    return results
