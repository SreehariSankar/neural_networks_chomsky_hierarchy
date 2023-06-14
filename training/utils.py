# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides utility functions for training and evaluation."""

import inspect
from typing import Callable

import chex
from jax import lax
from jax import nn as jnn
from jax import numpy as jnp

from neural_networks_chomsky_hierarchy.tasks import task

COMPUTATION_EMPTY_TOKEN = 0
OUTPUT_EMPTY_TOKEN = 1
DELIMITER_TOKEN = 2


def pad_input_tape(
    input_tape: chex.Array,
    generalization_task: task.GeneralizationTask,
    use_delimiters: bool = False,
    computation_steps_mult: int = 0,
) -> chex.Array:
  """Pads the input tape to account for the task's output length.

  For a given input tape `input_tape` of vocabulary size `vocab_sizes`, the
  padded tape will have the format [`delimiter_symbol`, `input_tape`,
  `empty_tape`, `delimiter_symbol`], where the empty tape corresponds to token
  `vocab_size + 1` and the `delimiter_symbol` corresponds to token `vocab_size +
  2`. The `empty_tape` has the same length as the task output.

  Args:
    input_tape: Input tokens of shape `(batch_size, input_length, input_size)`.
    generalization_task: The task that we train on.
    use_delimiters: Whether to use delimiter symbols on the tape.
    computation_steps_mult: The amount of empty cells to append to the input
      tape. This variable is a multiplier and the actual number of cell is
      computation_steps_mult * input_length.

  Returns:
    The padded sequence.
  """
  batch_size, input_length, input_size = input_tape.shape
  output_length = generalization_task.output_length(input_length)
  extra_dims_onehot = 3 if use_delimiters else 2

  # Adding extra zeros to the inputs to let space for the extra tokens.
  input_tape = jnp.concatenate(
      [input_tape,
       jnp.zeros((batch_size, input_length, extra_dims_onehot))],
      axis=-1)

  if computation_steps_mult > 0:
    # We create the portion of the tape used for computation. It contains a
    # first type of empty symbols.
    computation_tape = jnp.full(
        (batch_size, computation_steps_mult * input_length),
        fill_value=input_size + COMPUTATION_EMPTY_TOKEN)
    computation_tape = jnn.one_hot(
        computation_tape, num_classes=input_size + extra_dims_onehot)
    input_tape = jnp.concatenate([input_tape, computation_tape], axis=1)

  # Here we create the portion of the tape used to retrieve the output after
  # computation. It contains a second type of empty symbols.
  output_tape = jnp.full((batch_size, output_length),
                         fill_value=input_size + OUTPUT_EMPTY_TOKEN)
  output_tape = jnn.one_hot(
      output_tape, num_classes=input_size + extra_dims_onehot)
  input_tape = jnp.concatenate([input_tape, output_tape], axis=1)

  if use_delimiters:
    # We add the delimiter symbol to the beginning and the end of the tape.
    delimiter_symbol = jnp.full((batch_size, 1),
                                fill_value=input_size + DELIMITER_TOKEN)
    delimiter_symbol = jnn.one_hot(
        delimiter_symbol, num_classes=input_size + extra_dims_onehot)
    input_tape = jnp.concatenate(
        [delimiter_symbol, input_tape, delimiter_symbol], axis=1)
  return input_tape


def wrap_model_with_pad(
    model: Callable[..., chex.Array],
    generalization_task: task.GeneralizationTask,
    use_delimiters: bool = False,
    computation_steps_mult: int = 0,
    single_output: bool = False,
    is_autoregressive: bool = False,
) -> Callable[..., chex.Array]:
  """Returns a wrapped model that pads the inputs to match the output length.

  For a given input tape `input_tape` of vocabulary size `vocab_size`, the
  wrapped model will process a tape of the format
  [`delimiter_symbol`, `input_tape`, `empty_tape`, `delimiter_symbol`], where
  the empty tape token is `vocab_size + 1` and the `delimiter_symbol` token is
  `vocab_size + 2`. The `empty_tape` has the same length as the task output.

  Args:
    model: A model function that converts inputs to outputs.
    generalization_task: The task that we train on.
    use_delimiters: Whether to use delimiter symbols on the tape.
    computation_steps_mult: The amount of empty cells to append to the input
      tape. This variable is a multiplier and the actual number of cells is
      `computation_steps_mult * input_length`.
    single_output: Whether to return the squeezed tensor of values.
    is_autoregressive: Whether the model is autoregressive or not.
  """

  def new_model(x: chex.Array) -> chex.Array:
    output_length = generalization_task.output_length(x.shape[1])
    input_tape = pad_input_tape(x, generalization_task, use_delimiters,
                                computation_steps_mult)

    if 'input_length' in inspect.getfullargspec(model).args:
      output = model(input_tape, input_length=x.shape[1])
    else:
      output = model(input_tape)

    if use_delimiters:
      output = output[:, -output_length - 1:-1, :]
    output = output[:, -output_length:, :]

    if single_output:
      output = jnp.squeeze(output, axis=1)
    return output

  def new_model_autoregressive(
      x: chex.Array,
      y: chex.Array,
      sample: bool,
  ) -> chex.Array:
    """Returns an autoregressive model if `sample == True and output_size > 1`.

    Args:
      x: The input sequences of shape (b, t, i), where i is the input size.
      y: The target sequences of shape (b, t, o), where o is the output size.
      sample: Whether to evaluate the model using autoregressive decoding.
    """
    output_length = generalization_task.output_length(x.shape[1])

    if not sample or output_length == 1:
      output = model(x, y)

    else:

      def evaluate_model_autoregressively(
          idx: int,
          predictions: chex.Array,
      ) -> chex.Array:
        """Iteratively evaluates the model based on the previous predictions.

        Args:
          idx: The index of the target sequence that should be evaluated.
          predictions: The logits for the predictions up to but not including
            the index `idx`.

        Returns:
          The `predictions` array modified only at position `idx` where the
          logits for index `idx` have been inserted.
        """
        one_hot_predictions = jnn.one_hot(
            jnp.argmax(predictions, axis=-1),
            num_classes=generalization_task.output_size,
        )
        logits = model(x, one_hot_predictions)
        return predictions.at[:, idx].set(logits[:, idx])

      output = lax.fori_loop(
          lower=0,
          upper=output_length,
          body_fun=evaluate_model_autoregressively,
          init_val=jnp.empty_like(y))

    if single_output:
      output = jnp.squeeze(output, axis=1)
    return output

  if is_autoregressive:
    return new_model_autoregressive

  return new_model
