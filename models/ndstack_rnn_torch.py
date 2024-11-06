"""Non-deterministic Stack RNN core refactored for PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NDStack(nn.Module):
    """The non-deterministic stack."""

    def __init__(self, batch_size, stack_size, stack_states, stack_symbols):
        super(NDStack, self).__init__()
        self.gamma = torch.zeros(batch_size, stack_size, stack_size, stack_states, stack_symbols, stack_states, stack_symbols)
        self.alpha = torch.zeros(batch_size, stack_size, stack_states, stack_symbols)
        self.top_stack = torch.zeros(batch_size, stack_symbols)

    def update_stack(self, push_actions, pop_actions, replace_actions, timestep, read_states=True):
        stack_size = self.gamma.shape[2]
        mask = torch.zeros((stack_size, stack_size))
        mask[timestep - 1, timestep] = 1

        new_push_gamma_t = torch.einsum('bqxry,tT->btTqxry', push_actions, mask)[..., timestep]

        index_k = torch.arange(start=0, stop=stack_size).unsqueeze(0).repeat(stack_size, 1)
        index_i = index_k.T
        timestep_arr = torch.full((stack_size, stack_size), timestep)

        index_mask = (index_k > index_i) & (index_k < timestep_arr - 1)
        index_mask = torch.einsum('tT,bqxry->btTqxry', index_mask.float(), torch.ones_like(push_actions))

        new_pop_gamma_t = torch.einsum('bikqxuy,bkuysz,bszr->biqxry', index_mask * self.gamma, self.gamma[..., timestep - 1], pop_actions)

        new_replace_gamma_t = torch.einsum('biqxsz,bszry->biqxry', self.gamma[..., timestep - 1], replace_actions)

        self.gamma[:, timestep] = new_replace_gamma_t + new_pop_gamma_t + new_push_gamma_t

        alpha_t = torch.einsum('biqx,biqxry->bry', self.alpha, self.gamma[..., timestep])
        self.alpha[:, timestep] = alpha_t

        if read_states:
            batch_size, states, symbols = alpha_t.shape
            obs = alpha_t.view(batch_size, states * symbols)
        else:
            obs = alpha_t.sum(dim=1)

        obs = obs / (obs.sum(dim=-1, keepdim=True) + 1e-5)  # epsilon for numerical stability
        self.top_stack = obs
        return self


class NDStackRNNCore(nn.Module):
    """Core for the non-deterministic stack RNN using PyTorch."""

    def __init__(self, stack_symbols, stack_states, stack_size=30, inner_core=nn.RNNCell, read_states=False, **inner_core_kwargs):
        super(NDStackRNNCore, self).__init__()
        self.rnn_core = inner_core(**inner_core_kwargs)
        self.stack_symbols = stack_symbols
        self.stack_states = stack_states
        self.stack_size = stack_size
        self.read_states = read_states
        self.linear = nn.Linear(stack_states * stack_symbols**2 * 3, stack_states**2 * stack_symbols)

    def forward(self, inputs, prev_state):
        """Steps the non-deterministic stack RNN core."""
        ndstack, timestep, old_core_state = prev_state
        batch_size = ndstack.gamma.shape[0]

        inputs = torch.cat([inputs, ndstack.top_stack], dim=-1)
        new_core_output, new_core_state = self.rnn_core(inputs, old_core_state)

        n_push_actions = (self.stack_states * self.stack_symbols)**2
        n_pop_actions = self.stack_states**2 * self.stack_symbols
        n_replace_actions = n_push_actions

        actions = self.linear(new_core_output)
        actions = F.softmax(actions, dim=-1)

        push_actions = actions[:, :n_push_actions].view(batch_size, self.stack_states, self.stack_symbols, self.stack_states, self.stack_symbols)
        pop_actions = actions[:, n_push_actions:n_push_actions + n_pop_actions].view(batch_size, self.stack_states, self.stack_symbols, self.stack_states)
        replace_actions = actions[:, -n_replace_actions:].view(batch_size, self.stack_states, self.stack_symbols, self.stack_states, self.stack_symbols)

        ndstack = ndstack.update_stack(push_actions, pop_actions, replace_actions, timestep + 1, read_states=self.read_states)

        return new_core_output, (ndstack, timestep + 1, new_core_state)

    def initial_state(self, batch_size):
        """Returns the initial state of the core, a hidden state, and an empty stack."""
        core_state = self.rnn_core.weight_ih.data.new_zeros((batch_size, self.rnn_core.hidden_size))

        # Initialize gamma and alpha (transition matrix and node states).
        gamma = torch.zeros(batch_size, self.stack_size, self.stack_size, self.stack_states, self.stack_symbols, self.stack_states, self.stack_symbols)
        alpha = torch.zeros(batch_size, self.stack_size, self.stack_states, self.stack_symbols)
        alpha[:, 0, 0, 0] = 1  # Initial state in the stack

        if self.read_states:
            top_stack = torch.zeros(batch_size, self.stack_states * self.stack_symbols)
        else:
            top_stack = torch.zeros(batch_size, self.stack_symbols)

        ndstack = NDStack(gamma=gamma, alpha=alpha, top_stack=top_stack)
        return ndstack, torch.zeros((batch_size,), dtype=torch.int32), core_state
