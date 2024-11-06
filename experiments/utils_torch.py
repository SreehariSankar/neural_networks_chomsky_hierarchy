"""Provides utility functions for training and evaluation."""

import inspect
from typing import Any, Callable
import numpy as np

COMPUTATION_EMPTY_TOKEN = 0
OUTPUT_EMPTY_TOKEN = 1


def make_model_with_empty_targets(
        model: Callable[[np.ndarray], np.ndarray],
        generalization_task: Any,
        computation_steps_mult: int = 0,
        single_output: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a wrapped model that pads inputs to match output length."""

    def new_model(x: np.ndarray) -> np.ndarray:
        batch_size, input_length, input_size = x.shape
        output_length = generalization_task.output_length(input_length)
        extra_dims_onehot = 1 + int(computation_steps_mult > 0)
        final_input_size = input_size + extra_dims_onehot

        # Add trailing zeros to match final input size
        extra_zeros_x = np.zeros((batch_size, input_length, final_input_size - input_size))
        x = np.concatenate([x, extra_zeros_x], axis=-1)

        computation_tape = np.full(
            (batch_size, computation_steps_mult * input_length),
            fill_value=input_size + COMPUTATION_EMPTY_TOKEN
        )
        computation_tape = np.eye(final_input_size)[computation_tape]

        output_tokens = np.full(
            (batch_size, output_length),
            fill_value=input_size + OUTPUT_EMPTY_TOKEN - int(computation_steps_mult == 0)
        )
        output_tokens = np.eye(final_input_size)[output_tokens]

        final_input = np.concatenate([x, computation_tape, output_tokens], axis=1)

        if 'input_length' in inspect.getfullargspec(model).args:
            output = model(final_input, input_length=input_length)
        else:
            output = model(final_input)

        output = output[:, -output_length:]
        if single_output:
            output = np.squeeze(output, axis=1)
        return output

    return new_model


def make_model_with_targets_as_input(
        model: Callable[[np.ndarray], np.ndarray], computation_steps_mult: int = 0
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Returns a wrapped model that takes targets as input, useful for autoregressive models."""

    def new_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size, input_length, input_size = x.shape
        _, output_length, output_size = y.shape
        extra_dims_onehot = 1 + int(computation_steps_mult > 0)
        final_input_size = max(input_size, output_size) + extra_dims_onehot

        extra_zeros_x = np.zeros((batch_size, input_length, final_input_size - input_size))
        x = np.concatenate([x, extra_zeros_x], axis=-1)

        extra_zeros_y = np.zeros((batch_size, output_length, final_input_size - output_size))
        y = np.concatenate([y, extra_zeros_y], axis=-1)

        computation_tape = np.full(
            (batch_size, computation_steps_mult * input_length),
            fill_value=input_size + COMPUTATION_EMPTY_TOKEN
        )
        computation_tape = np.eye(final_input_size)[computation_tape]

        output_token = np.full(
            (batch_size, 1),
            fill_value=input_size + OUTPUT_EMPTY_TOKEN - int(computation_steps_mult == 0)
        )
        output_token = np.eye(final_input_size)[output_token]
        final_input = np.concatenate([x, computation_tape, output_token, y], axis=1)

        if 'input_length' in inspect.getfullargspec(model).args:
            output = model(final_input, input_length=input_length)
        else:
            output = model(final_input)

        return output[:, -output_length - 1: -1]

    return new_model


def add_sampling_to_autoregressive_model(
        model: Callable[[np.ndarray, np.ndarray], np.ndarray],
        single_output: bool = False,
) -> Callable[[np.ndarray, np.ndarray, bool], np.ndarray]:
    """Adds a 'sample' argument for autoregressive sampling in the model."""

    def new_model_with_sampling(
            x: np.ndarray,
            y: np.ndarray,
            sample: bool,
    ) -> np.ndarray:
        output_length = 1 if len(y.shape) == 2 else y.shape[1]
        output_size = y.shape[-1]

        if not sample or output_length == 1:
            return model(x, y)

        predictions = np.empty_like(y)

        for idx in range(output_length):
            one_hot_predictions = np.eye(output_size)[np.argmax(predictions, axis=-1)]
            logits = model(x, one_hot_predictions)
            predictions[:, idx] = logits[:, idx]

        if single_output:
            predictions = np.squeeze(predictions, axis=1)
        return predictions

    return new_model_with_sampling


def update_tree_with_new_containers(
        tree: Any, update_dict: dict[str, Any]
) -> None:
    """Updates a dataclass tree in place, adding new containers."""

    for key, value in update_dict.items():
        subkeys = key.split('.')
        target = tree

        # Traverse to the last container
        for i, subkey in enumerate(subkeys[:-1]):
            if not hasattr(target, subkey):
                setattr(target, subkey, {})
            target = getattr(target, subkey)

        setattr(target, subkeys[-1], value)
