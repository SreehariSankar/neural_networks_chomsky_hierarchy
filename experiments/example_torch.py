"""Example script to train and evaluate a network."""

import argparse
import numpy as np

from neural_networks_chomsky_hierarchy.experiments import constants_torch as constants
from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.experiments import training_torch as training
from neural_networks_chomsky_hierarchy.experiments import utils_torch as utils

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a network.")
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--sequence_length', type=int, default=40, help='Maximum training sequence length.')
    parser.add_argument('--task', type=str, default='even_pairs', help='Length generalization task (see constants.py).')
    parser.add_argument('--architecture', type=str, default='tape_rnn', help='Model architecture (see constants.py).')
    parser.add_argument('--is_autoregressive', action='store_true', help='Use autoregressive sampling if set.')
    parser.add_argument('--computation_steps_mult', type=int, default=0, help='Computation steps multiplier.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Define architecture parameters (can be adjusted as needed)
    architecture_params = {
        'hidden_size': 256,
        'memory_cell_size': 8,
        'memory_size': 40,
    }

    # Create the task and curriculum
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(1, args.sequence_length + 1))
    )
    task = constants.TASK_BUILDERS[args.task]()

    # Create the model
    single_output = task.output_length(10) == 1
    model = constants.MODEL_BUILDERS[args.architecture](
        output_size=task.output_size,
        return_all_outputs=True,
        **architecture_params,
    )

    if args.is_autoregressive:
        if 'transformer' not in args.architecture:
            model = utils.make_model_with_targets_as_input(
                model, args.computation_steps_mult
            )
        model = utils.add_sampling_to_autoregressive_model(model, single_output)
    else:
        model = utils.make_model_with_empty_targets(
            model, task, args.computation_steps_mult, single_output
        )

    # Define loss and accuracy functions
    def loss_fn(output, target):
        loss = np.mean(np.sum(task.pointwise_loss_fn(output, target), axis=-1))
        return loss, {}

    def accuracy_fn(output, target):
        mask = task.accuracy_mask(target)
        return np.sum(mask * task.accuracy_fn(output, target)) / np.sum(mask)

    # Set up training parameters
    training_params = training.ClassicTrainingParams(
        seed=0,
        model_init_seed=0,
        training_steps=10_000,
        log_frequency=100,
        length_curriculum=curriculum,
        batch_size=args.batch_size,
        task=task,
        model=model,
        loss_fn=loss_fn,
        learning_rate=1e-3,
        accuracy_fn=accuracy_fn,
        compute_full_range_test=True,
        max_range_test_length=100,
        range_test_total_batch_size=512,
        range_test_sub_batch_size=64,
        is_autoregressive=args.is_autoregressive,
    )

    # Run the training and evaluation
    training_worker = training.TrainingWorker(training_params, use_tqdm=True)
    _, eval_results, _ = training_worker.run()

    # Gather and print results
    accuracies = [r['accuracy'] for r in eval_results]
    score = np.mean(accuracies[args.sequence_length + 1:])
    print(f'Network score: {score}')


if __name__ == '__main__':
    main()
