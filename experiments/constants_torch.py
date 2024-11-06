"""Constants for our length generalization experiments."""

import functools

import torch.nn as nn

from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.models import ndstack_rnn_torch as ndstack_rnn
from neural_networks_chomsky_hierarchy.models import rnn_torch as rnn
from neural_networks_chomsky_hierarchy.models import stack_rnn_torch as stack_rnn
from neural_networks_chomsky_hierarchy.models import tape_rnn_torch as tape_rnn
from neural_networks_chomsky_hierarchy.models import transformer_torch as transformer
from neural_networks_chomsky_hierarchy.tasks.cs import binary_addition
from neural_networks_chomsky_hierarchy.tasks.cs import binary_multiplication
from neural_networks_chomsky_hierarchy.tasks.cs import bucket_sort
from neural_networks_chomsky_hierarchy.tasks.cs import compute_sqrt
from neural_networks_chomsky_hierarchy.tasks.cs import duplicate_string
from neural_networks_chomsky_hierarchy.tasks.cs import missing_duplicate_string
from neural_networks_chomsky_hierarchy.tasks.cs import odds_first
from neural_networks_chomsky_hierarchy.tasks.dcf import modular_arithmetic_brackets
from neural_networks_chomsky_hierarchy.tasks.dcf import reverse_string
from neural_networks_chomsky_hierarchy.tasks.dcf import solve_equation
from neural_networks_chomsky_hierarchy.tasks.dcf import stack_manipulation
from neural_networks_chomsky_hierarchy.tasks.regular import cycle_navigation
from neural_networks_chomsky_hierarchy.tasks.regular import even_pairs
from neural_networks_chomsky_hierarchy.tasks.regular import modular_arithmetic
from neural_networks_chomsky_hierarchy.tasks.regular import parity_check


MODEL_BUILDERS = {
    'rnn':
        functools.partial(rnn.make_rnn, rnn_core=nn.RNN),
    'lstm':
        functools.partial(rnn.make_rnn, rnn_core=nn.LSTM),
    'stack_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=stack_rnn.StackRNNCore,
            inner_core=nn.RNN),
    'ndstack_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=ndstack_rnn.NDStackRNNCore,
            inner_core=nn.RNN),
    'stack_lstm':
        functools.partial(
            rnn.make_rnn, rnn_core=stack_rnn.StackRNNCore, inner_core=nn.LSTM),
    'transformer_encoder':
        transformer.make_transformer_encoder,
    'transformer':
        transformer.make_transformer,
    'tape_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=tape_rnn.TapeInputLengthJumpCore,
            inner_core=nn.RNN),
}

CURRICULUM_BUILDERS = {
    'fixed': curriculum_lib.FixedCurriculum,
    'regular_increase': curriculum_lib.RegularIncreaseCurriculum,
    'reverse_exponential': curriculum_lib.ReverseExponentialCurriculum,
    'uniform': curriculum_lib.UniformCurriculum,
}

TASK_BUILDERS = {
    'modular_arithmetic':
        modular_arithmetic.ModularArithmetic,
    'parity_check':
        parity_check.ParityCheck,
    'even_pairs':
        even_pairs.EvenPairs,
    'cycle_navigation':
        cycle_navigation.CycleNavigation,
    'modular_arithmetic_brackets':
        functools.partial(
            modular_arithmetic_brackets.ModularArithmeticBrackets, mult=True),
    'reverse_string':
        reverse_string.ReverseString,
    'missing_duplicate_string':
        missing_duplicate_string.MissingDuplicateString,
    'duplicate_string':
        duplicate_string.DuplicateString,
    'binary_addition':
        binary_addition.BinaryAddition,
    'binary_multiplication':
        binary_multiplication.BinaryMultiplication,
    'compute_sqrt':
        compute_sqrt.ComputeSqrt,
    'odds_first':
        odds_first.OddsFirst,
    'solve_equation':
        solve_equation.SolveEquation,
    'stack_manipulation':
        stack_manipulation.StackManipulation,
    'bucket_sort':
        bucket_sort.BucketSort,
}

TASK_LEVELS = {
    'modular_arithmetic': 'regular',
    'parity_check': 'regular',
    'even_pairs': 'regular',
    'cycle_navigation': 'regular',
    'modular_arithmetic_brackets': 'dcf',
    'reverse_string': 'dcf',
    'stack_manipulation': 'dcf',
    'solve_equation': 'dcf',
    'missing_duplicate_string': 'cs',
    'compute_sqrt': 'cs',
    'duplicate_string': 'cs',
    'binary_addition': 'cs',
    'binary_multiplication': 'cs',
    'odds_first': 'cs',
    'bucket_sort': 'cs',
}
"""Constants for our length generalization experiments."""

import functools

import torch.nn as nn

from neural_networks_chomsky_hierarchy.experiments import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.models import ndstack_rnn_torch as ndstack_rnn
from neural_networks_chomsky_hierarchy.models import rnn_torch as rnn
from neural_networks_chomsky_hierarchy.models import stack_rnn_torch as stack_rnn
from neural_networks_chomsky_hierarchy.models import tape_rnn_torch as tape_rnn
from neural_networks_chomsky_hierarchy.models import transformer_torch as transformer
from neural_networks_chomsky_hierarchy.tasks.cs import binary_addition
from neural_networks_chomsky_hierarchy.tasks.cs import binary_multiplication
from neural_networks_chomsky_hierarchy.tasks.cs import bucket_sort
from neural_networks_chomsky_hierarchy.tasks.cs import compute_sqrt
from neural_networks_chomsky_hierarchy.tasks.cs import duplicate_string
from neural_networks_chomsky_hierarchy.tasks.cs import missing_duplicate_string
from neural_networks_chomsky_hierarchy.tasks.cs import odds_first
from neural_networks_chomsky_hierarchy.tasks.dcf import modular_arithmetic_brackets
from neural_networks_chomsky_hierarchy.tasks.dcf import reverse_string
from neural_networks_chomsky_hierarchy.tasks.dcf import solve_equation
from neural_networks_chomsky_hierarchy.tasks.dcf import stack_manipulation
from neural_networks_chomsky_hierarchy.tasks.regular import cycle_navigation
from neural_networks_chomsky_hierarchy.tasks.regular import even_pairs
from neural_networks_chomsky_hierarchy.tasks.regular import modular_arithmetic
from neural_networks_chomsky_hierarchy.tasks.regular import parity_check


MODEL_BUILDERS = {
    'rnn':
        functools.partial(rnn.make_rnn, rnn_core=nn.RNN),
    'lstm':
        functools.partial(rnn.make_rnn, rnn_core=nn.LSTM),
    'stack_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=stack_rnn.StackRNNCore,
            inner_core=nn.RNN),
    'ndstack_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=ndstack_rnn.NDStackRNNCore,
            inner_core=nn.RNN),
    'stack_lstm':
        functools.partial(
            rnn.make_rnn, rnn_core=stack_rnn.StackRNNCore, inner_core=nn.LSTM),
    'transformer_encoder':
        transformer.make_transformer_encoder,
    'transformer':
        transformer.make_transformer,
    'tape_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=tape_rnn.TapeInputLengthJumpCore,
            inner_core=nn.RNN),
}

CURRICULUM_BUILDERS = {
    'fixed': curriculum_lib.FixedCurriculum,
    'regular_increase': curriculum_lib.RegularIncreaseCurriculum,
    'reverse_exponential': curriculum_lib.ReverseExponentialCurriculum,
    'uniform': curriculum_lib.UniformCurriculum,
}

TASK_BUILDERS = {
    'modular_arithmetic':
        modular_arithmetic.ModularArithmetic,
    'parity_check':
        parity_check.ParityCheck,
    'even_pairs':
        even_pairs.EvenPairs,
    'cycle_navigation':
        cycle_navigation.CycleNavigation,
    'modular_arithmetic_brackets':
        functools.partial(
            modular_arithmetic_brackets.ModularArithmeticBrackets, mult=True),
    'reverse_string':
        reverse_string.ReverseString,
    'missing_duplicate_string':
        missing_duplicate_string.MissingDuplicateString,
    'duplicate_string':
        duplicate_string.DuplicateString,
    'binary_addition':
        binary_addition.BinaryAddition,
    'binary_multiplication':
        binary_multiplication.BinaryMultiplication,
    'compute_sqrt':
        compute_sqrt.ComputeSqrt,
    'odds_first':
        odds_first.OddsFirst,
    'solve_equation':
        solve_equation.SolveEquation,
    'stack_manipulation':
        stack_manipulation.StackManipulation,
    'bucket_sort':
        bucket_sort.BucketSort,
}

TASK_LEVELS = {
    'modular_arithmetic': 'regular',
    'parity_check': 'regular',
    'even_pairs': 'regular',
    'cycle_navigation': 'regular',
    'modular_arithmetic_brackets': 'dcf',
    'reverse_string': 'dcf',
    'stack_manipulation': 'dcf',
    'solve_equation': 'dcf',
    'missing_duplicate_string': 'cs',
    'compute_sqrt': 'cs',
    'duplicate_string': 'cs',
    'binary_addition': 'cs',
    'binary_multiplication': 'cs',
    'odds_first': 'cs',
    'bucket_sort': 'cs',
}
