import torch
import torch.nn as nn
import torch.nn.functional as F

_NUM_ACTIONS = 3  # Number of actions: POP, PUSH, NO_OP

# Custom RNN state, which includes stacks and hidden internal state
_StackRnnState = tuple[torch.Tensor, torch.Tensor]


def _update_stack(stack: torch.Tensor, actions: torch.Tensor, push_value: torch.Tensor) -> torch.Tensor:
    """Updates the stack values."""
    batch_size, stack_size, stack_cell_size = stack.shape

    # Reshape and expand the actions to match the stack size
    cell_tiled_stack_actions = actions.unsqueeze(-1).repeat(1, 1, stack_cell_size)

    push_action = cell_tiled_stack_actions[..., 0]
    pop_action = cell_tiled_stack_actions[..., 1]
    no_op_action = cell_tiled_stack_actions[..., 2]

    pop_value = stack[:, 1, :]
    no_op_value = stack[:, 0, :]

    # Update the top of the stack
    top_new_stack = (
            push_action * push_value + pop_action * pop_value + no_op_action * no_op_value
    ).unsqueeze(1)

    # Update the rest of the stack
    stack_tiled_stack_actions = actions.unsqueeze(1).repeat(1, stack_size - 1, 1, stack_cell_size)
    push_action = stack_tiled_stack_actions[..., 0]
    pop_action = stack_tiled_stack_actions[..., 1]
    no_op_action = stack_tiled_stack_actions[..., 2]

    push_value = stack[:, :-1, :]
    pop_extra_zeros = torch.zeros((batch_size, 1, stack_cell_size), device=stack.device)
    pop_value = torch.cat([stack[:, 2:, :], pop_extra_zeros], dim=1)
    no_op_value = stack[:, 1:, :]

    rest_new_stack = (
            push_action * push_value + pop_action * pop_value + no_op_action * no_op_value
    )

    # Concatenate the top of the stack with the rest
    return torch.cat([top_new_stack, rest_new_stack], dim=1)


class StackRNNCore(nn.Module):
    """Core for the stack RNN in PyTorch."""

    def __init__(self, stack_cell_size: int, stack_size: int = 30, n_stacks: int = 1, inner_core: type[nn.Module] = nn.RNNCell, **inner_core_kwargs):
        """Initializes the StackRNNCore.

        Args:
            stack_cell_size: The dimension of the vectors we put in the stack.
            stack_size: The total number of vectors we can stack.
            n_stacks: Number of stacks to use in the network.
            inner_core: The inner RNN core, such as `nn.RNNCell`.
            **inner_core_kwargs: Additional arguments to be passed to the RNN core.
        """
        super(StackRNNCore, self).__init__()
        self.inner_core = inner_core(**inner_core_kwargs)
        self.stack_cell_size = stack_cell_size
        self.stack_size = stack_size
        self.n_stacks = n_stacks

        # Linear layers for the push values and actions
        self.push_value_linear = nn.Linear(self.inner_core.hidden_size, n_stacks * stack_cell_size)
        self.stack_action_linear = nn.Linear(self.inner_core.hidden_size, n_stacks * _NUM_ACTIONS)

    def forward(self, inputs: torch.Tensor, prev_state: _StackRnnState) -> tuple[torch.Tensor, _StackRnnState]:
        """Steps the stack RNN core."""
        stacks, old_core_state = prev_state

        # The network can always read the top of the stack
        batch_size = stacks.shape[0]
        top_stacks = stacks[:, :, 0, :]
        top_stacks = top_stacks.view(batch_size, -1)

        inputs = torch.cat([inputs, top_stacks], dim=-1)
        new_core_output = self.inner_core(inputs, old_core_state)

        # Compute push values and actions
        push_values = self.push_value_linear(new_core_output)
        push_values = push_values.view(batch_size, self.n_stacks, self.stack_cell_size)

        stack_actions = self.stack_action_linear(new_core_output)
        stack_actions = F.softmax(stack_actions, dim=-1)
        stack_actions = stack_actions.view(batch_size, self.n_stacks, _NUM_ACTIONS)

        # Update the stack
        new_stacks = torch.stack(
            [_update_stack(stack, action, push_value) for stack, action, push_value in zip(stacks, stack_actions, push_values)], dim=0
        )

        # No separate new_core_state for RNNCell
        return new_core_output, (new_stacks, new_core_output)

    def initial_state(self, batch_size: int) -> _StackRnnState:
        """Returns the initial state of the core, including the empty stack and the initial RNN state."""
        core_state = torch.zeros((batch_size, self.inner_core.hidden_size), device=self.push_value_linear.weight.device)
        stacks = torch.zeros(
            (batch_size, self.n_stacks, self.stack_size, self.stack_cell_size), device=self.push_value_linear.weight.device
        )
        return stacks, core_state

