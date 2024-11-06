"""Training loop for length generalization experiments."""

import dataclasses
import random
from typing import Any, Callable, Mapping, Optional
import numpy as np
from tqdm import tqdm

from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.experiments import range_evaluation_torch as range_evaluation
from neural_networks_chomsky_hierarchy.tasks import task_torch as task_lib

# Type aliases for readability
LossMetrics = Optional[Mapping[str, np.ndarray]]
LossFn = Callable[[np.ndarray, np.ndarray], tuple[float, LossMetrics]]
AccuracyFn = Callable[[np.ndarray, np.ndarray], float]


@dataclasses.dataclass
class ClassicTrainingParams:
    """Parameters needed to train classical architectures."""
    seed: int  # Seed for reproducibility.
    model_init_seed: int  # Seed to initialize model parameters.
    training_steps: int
    log_frequency: int

    task: task_lib.GeneralizationTask
    length_curriculum: curriculum_lib.Curriculum
    batch_size: int

    model: Callable  # Model initialization function.
    loss_fn: LossFn
    learning_rate: float
    max_grad_norm: float = 1.0
    is_autoregressive: bool = False

    compute_full_range_test: bool = False
    range_test_total_batch_size: int = 512
    range_test_sub_batch_size: int = 64
    max_range_test_length: int = 100

    accuracy_fn: Optional[AccuracyFn] = None


def apply_loss_and_metrics(
        params: Any,
        batch: task_lib.Batch,
        model_apply_fn: Callable,
        loss_fn: LossFn,
        accuracy_fn: Optional[AccuracyFn] = None,
        is_autoregressive: bool = False,
) -> tuple[float, tuple[LossMetrics, float]]:
    """Computes the model output and applies the loss function."""
    if is_autoregressive:
        outputs = model_apply_fn(params, batch["input"], batch["output"], sample=False)
    else:
        outputs = model_apply_fn(params, batch["input"])

    loss, loss_metrics = loss_fn(outputs, batch["output"])
    accuracy = accuracy_fn(outputs, batch["output"]) if accuracy_fn else None
    return loss, (loss_metrics, accuracy)


def update_parameters(
        params: Any,
        batch: task_lib.Batch,
        model_apply_fn: Callable,
        loss_fn: LossFn,
        accuracy_fn: Optional[AccuracyFn],
        optimizer: Callable,
        opt_state: Any,
        is_autoregressive: bool = False,
) -> tuple[Any, Any, tuple[float, LossMetrics, float]]:
    """Applies a single update step to the model parameters."""
    loss, (metrics, accuracy) = apply_loss_and_metrics(
        params, batch, model_apply_fn, loss_fn, accuracy_fn, is_autoregressive
    )
    grads = np.gradient(loss)  # Placeholder for gradient calculation logic
    updates = optimizer(grads, params, opt_state)
    params = params - updates  # Placeholder for parameter update logic
    return params, opt_state, (loss, metrics, accuracy)


class TrainingWorker:
    """Worker to handle the training process."""

    def __init__(self, training_params: ClassicTrainingParams, use_tqdm: bool = False):
        """
        Args:
            training_params: The training parameters.
            use_tqdm: Whether to add a progress bar to stdout.
        """
        self._training_params = training_params
        self._use_tqdm = use_tqdm
        self._params = None
        self._step = 0

    def run(self) -> tuple[list[Mapping[str, Any]], Optional[list[Mapping[str, Any]]], Any]:
        """Trains the model with the provided configuration."""
        training_params = self._training_params
        random.seed(training_params.seed)
        np.random.seed(training_params.seed)

        # Initialize model, task, curriculum, and optimizer
        results = []
        task = training_params.task
        length_curriculum = training_params.length_curriculum

        dummy_batch = task.sample_batch(length=10, batch_size=training_params.batch_size)
        self._params = training_params.model.init_params(dummy_batch["input"])
        opt_state = np.zeros_like(self._params)  # Placeholder for optimizer state

        # Training loop
        steps = range(training_params.training_steps + 1)
        if self._use_tqdm:
            steps = tqdm(steps)

        for step in steps:
            # Sample a batch and update model parameters
            length = length_curriculum.sample_sequence_length(step)
            train_batch = task.sample_batch(length=length, batch_size=training_params.batch_size)

            params, opt_state, (train_loss, train_metrics, train_accuracy) = update_parameters(
                params=self._params,
                batch=train_batch,
                model_apply_fn=training_params.model.apply,
                loss_fn=training_params.loss_fn,
                accuracy_fn=training_params.accuracy_fn,
                optimizer=self.simple_optimizer(training_params.learning_rate),
                opt_state=opt_state,
                is_autoregressive=training_params.is_autoregressive,
            )

            self._params, self._step = params, step

            # Logging
            if training_params.log_frequency > 0 and step % training_params.log_frequency == 0:
                log_data = {"step": step, "train_loss": float(train_loss)}
                if training_params.accuracy_fn:
                    log_data["train_accuracy"] = float(train_accuracy)
                for key, value in train_metrics.items():
                    log_data[f"train_metrics.{key}"] = value
                results.append(log_data)

        # Optional evaluation
        eval_results = None
        if training_params.compute_full_range_test:
            eval_results = self.evaluate_full_range(task)

        return results, eval_results, params

    @staticmethod
    def simple_optimizer(learning_rate: float):
        """Simple gradient descent optimizer."""

        def optimizer(grads, params, state):
            return -learning_rate * grads

        return optimizer

    def evaluate_full_range(self, task: task_lib.GeneralizationTask) -> list[Mapping[str, Any]]:
        """Runs full range evaluation if configured."""
        training_params = self._training_params
        eval_params = range_evaluation.EvaluationParams(
            model=training_params.model,
            params=self._params,
            accuracy_fn=training_params.accuracy_fn,
            sample_batch=task.sample_batch,
            max_test_length=training_params.max_range_test_length,
            total_batch_size=training_params.range_test_total_batch_size,
            sub_batch_size=training_params.range_test_sub_batch_size,
            is_autoregressive=training_params.is_autoregressive,
        )
        return range_evaluation.range_evaluation(eval_params, use_tqdm=False)
