

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Any, Optional, Sequence, Tuple


class TapeRNNCore(L.LightningModule):
    """Core for the tape RNN."""

    def __init__(
        self,
        memory_cell_size: int,
        memory_size: int = 30,
        n_tapes: int = 1,
        mlp_layers_size: Sequence[int] = (64, 64),
        inner_core: type = nn.RNN,
        name: Optional[str] = None,
        **inner_core_kwargs: Any
    ):
        """Initializes the TapeRNNCore.

        Args:
            memory_cell_size: The dimension of the vectors we put in memory.
            memory_size: The size of the tape, fixed value along the episode.
            n_tapes: Number of tapes to use. Default is 1.
            mlp_layers_size: Sizes for the inner MLP layers.
            inner_core: The inner RNN core class.
            name: Name for the module (optional).
            **inner_core_kwargs: Additional kwargs for the RNN core.
        """
        super().__init__()
        self.memory_cell_size = memory_cell_size
        self.memory_size = memory_size
        self.n_tapes = n_tapes

        # RNN core
        self.rnn_core = inner_core(**inner_core_kwargs)
        self.core_hidden_size = inner_core_kwargs.get("hidden_size", 64)

        # MLP for write values
        mlp_sizes = [self.core_hidden_size] + list(mlp_layers_size) + [n_tapes * memory_cell_size]
        self.mlp = nn.Sequential(*[nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]) for i in range(len(mlp_sizes) - 1)])

        # Action prediction layer
        self.action_layer = nn.Linear(self.core_hidden_size, self.num_actions * n_tapes)

    @property
    def num_actions(self) -> int:
        """Returns the number of actions which can be taken on the tape."""
        raise NotImplementedError("This property should be implemented in subclasses.")

    def forward(self, inputs: torch.Tensor, prev_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        memories, old_core_state, input_length = prev_state

        batch_size = memories.shape[0]
        current_memories = memories[:, :, 0, :].reshape(batch_size, -1)
        inputs = torch.cat([inputs, current_memories], dim=-1)

        # Run through the RNN core
        new_core_output, new_core_state = self.rnn_core(inputs, old_core_state)

        # Run through MLP for write values
        write_values = self.mlp(new_core_output).reshape(batch_size, self.n_tapes, self.memory_cell_size)

        # Calculate action probabilities
        actions = self.action_layer(new_core_output).reshape(batch_size, self.n_tapes, self.num_actions)
        actions = F.softmax(actions, dim=-1)

        # Update memory using actions and write values
        new_memories = self._update_memory(memories, actions, write_values, input_length)
        return new_core_output, (new_memories, new_core_state, input_length)

    def _update_memory(self, memory: torch.Tensor, actions: torch.Tensor, write_values: torch.Tensor, input_length: int) -> torch.Tensor:
        batch_size, memory_size, _ = memory.size()

        # Insert write values into memory
        memory_with_write = torch.cat([write_values.unsqueeze(2), memory[:, :, 1:]], dim=2)

        # Apply operations defined by `_tape_operations`
        eye_memory = torch.eye(memory_size, device=memory.device)
        operations = self._tape_operations(eye_memory, input_length)
        memory_operations = [torch.einsum('mk,bkc->bmc', op, memory_with_write) for op in operations]

        # Aggregate memory operations with action weights
        memory_operations = torch.stack(memory_operations, dim=1)
        new_memory = torch.einsum('bmoc,boa->bmc', memory_operations, actions)
        return new_memory

    def initial_state(self, batch_size: int, input_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        core_state = torch.zeros(batch_size, self.core_hidden_size, device=self.device)
        memories = torch.zeros(batch_size, self.n_tapes, self.memory_size, self.memory_cell_size, device=self.device)
        return memories, core_state, torch.tensor([input_length], device=self.device)


class TapeInputLengthJumpCore(TapeRNNCore):
    """A tape-RNN with extra jumps of the length of the input."""

    @property
    def num_actions(self) -> int:
        """Returns the number of actions of the tape."""
        return 5

    def _tape_operations(self, eye_memory: torch.Tensor, input_length: int) -> list:
        write_stay = eye_memory
        write_left = torch.roll(eye_memory, shifts=-1, dims=0)
        write_right = torch.roll(eye_memory, shifts=1, dims=0)
        write_jump_left = torch.roll(eye_memory, shifts=-input_length, dims=0)
        write_jump_right = torch.roll(eye_memory, shifts=input_length, dims=0)
        return [write_stay, write_left, write_right, write_jump_left, write_jump_right]
