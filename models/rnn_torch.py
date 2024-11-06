"""Builders for RNN/LSTM cores refactored for PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self, output_size: int, rnn_core: type[nn.Module], return_all_outputs: bool = False,
                 input_window: int = 1, **rnn_kwargs):
        """Initializes the RNN model.

        Args:
            output_size: The output size of the model.
            rnn_core: The PyTorch RNN core to use, e.g., nn.LSTM or nn.RNN.
            return_all_outputs: Whether to return the whole sequence of outputs or just the last one.
            input_window: The number of tokens that are fed at once to the RNN.
            **rnn_kwargs: Kwargs to be passed to the RNN core.
        """
        super(RNNModel, self).__init__()
        self.return_all_outputs = return_all_outputs
        self.input_window = input_window
        self.rnn_core = rnn_core(**rnn_kwargs)
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Linear(self.rnn_core.hidden_size, output_size)

    def forward(self, x: torch.Tensor, input_length: int = 1) -> torch.Tensor:
        batch_size, seq_length, embed_size = x.shape

        # Padding for input_window
        if seq_length % self.input_window != 0:
            padding_length = self.input_window - seq_length % self.input_window
            x = F.pad(x, (0, 0, 0, padding_length))  # Pad the second last dimension

        new_seq_length = x.shape[1]
        x = x.view(batch_size, new_seq_length // self.input_window, self.input_window * embed_size)

        # Flatten the input to feed into the RNN core
        x = self.flatten(x)

        # Initial hidden state
        h0 = torch.zeros(1, batch_size, self.rnn_core.hidden_size, device=x.device)

        # Forward through the RNN/LSTM core
        output, _ = self.rnn_core(x, h0)

        output = output.view(batch_size, new_seq_length, -1)

        if not self.return_all_outputs:
            output = output[:, -1, :]  # Only return the last time step

        output = F.relu(output)
        output = self.linear(output)

        return output
