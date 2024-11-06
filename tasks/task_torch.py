"""Base class for length generalization tasks."""

import abc
from typing import TypedDict
import numpy as np

Batch = TypedDict('Batch', {'input': np.ndarray, 'output': np.ndarray})


class GeneralizationTask(abc.ABC):
    """A task for the generalization project.

    Exposes a sample_batch method and details about input/output sizes,
    losses, and accuracies.
    """

    @abc.abstractmethod
    def sample_batch(self, rng: np.random.Generator, batch_size: int, length: int) -> Batch:
        """Returns a batch of inputs/outputs."""

    def pointwise_loss_fn(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns the pointwise loss between an output and a target."""
        return -target * np.log_softmax(output, axis=-1)

    def accuracy_fn(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns the accuracy between an output and a target."""
        return (np.argmax(output, axis=-1) == np.argmax(target, axis=-1)).astype(np.float32)

    def accuracy_mask(self, target: np.ndarray) -> np.ndarray:
        """Returns a mask to compute the accuracies, to remove the superfluous ones."""
        # Target shape is (B, T, C), where C is the number of classes.
        # Mask shape should be (B, T).
        return np.ones(target.shape[:-1])

    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        """Returns the size of the input of the models trained on this task."""

    @property
    @abc.abstractmethod
    def output_size(self) -> int:
        """Returns the size of the output of the models trained on this task."""

    def output_length(self, input_length: int) -> int:
        """Returns the length of the output, given an input length."""
        return 1
