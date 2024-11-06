"""Example script to train and evaluate a Transformer on the ModularArithmetic task."""

import argparse
import numpy as np
import torch
import torch.nn as nn

# Import the necessary components from the project structure
from neural_networks_chomsky_hierarchy.tasks.regular.modular_arithmetic import ModularArithmetic
from neural_networks_chomsky_hierarchy.models.transformer_torch import Transformer  # Transformer model
from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.experiments import training_torch as training
from neural_networks_chomsky_hierarchy.models.transformer_torch import TransformerConfig
from neural_networks_chomsky_hierarchy.experiments import utils_torch as utils
from neural_networks_chomsky_hierarchy.experiments import constants_torch as constants
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a Transformer on ModularArithmetic.")
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--sequence_length', type=int, default=9, help='Maximum training sequence length.')
    parser.add_argument('--modulus', type=int, default=5, help='Modulus for the ModularArithmetic task.')
    parser.add_argument('--model_dim', type=int, default=64, help='Transformer model dimension.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer layers.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--is_autoregressive', action='store_true', help='Use autoregressive sampling if set.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Define the ModularArithmetic task
    task = ModularArithmetic(modulus=args.modulus)
    curriculum = curriculum_lib.UniformCurriculum(values=list(range(1, args.sequence_length + 1)))

    # Define the Transformer model configuration
    config = TransformerConfig(
        embedding_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        output_size=task.output_size,
    )

    # Initialize the Transformer model
    model = Transformer(config).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Define loss and accuracy functions compatible with PyTorch
    def loss_fn(output, target):
        criterion = nn.CrossEntropyLoss()
        return criterion(output.view(-1, config.output_size), target.argmax(dim=-1).view(-1)), {}

    def accuracy_fn(output, target):
        predictions = torch.argmax(output, dim=-1)
        return (predictions == target.argmax(dim=-1)).float().mean().item()

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
