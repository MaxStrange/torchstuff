"""
Simple character-based RNN for classification of sequences.

From https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""
# pylint: disable=no-member
import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    Home-spun recurrent cell. Simply takes an input and previous hidden state
    at each step, concats them and runs the concatenated vector through
    two linear layers to get a next hidden state and a next output vector.
    The output vector is then run through a softmax layer to get the output vector.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Inputs
        ------

        * One-hot encoding of ASCII characters
        * Previous hidden state

        Outputs
        -------

        * Softmax over classes
        * Next hidden state

        Arguments
        ---------
        - input_size: The length of the one-hot encoding vector
        - hidden_size: The length of the hidden state vector
        - output_size: The number of classes

        """
        super(RNN, self).__init__()

        # Metadata
        self.hidden_size = hidden_size

        # Layers
        self.in_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in_to_output = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_: torch.tensor, hidden: torch.tensor):
        """
        Forward pass of the input and previous hidden state.
        """
        # Concatenate the input and hidden vectors
        combined = torch.cat((input_, hidden), dim=1)

        # Hidden state branch
        hidden = self.in_to_hidden(combined)

        # Output branch
        output = self.in_to_output(combined)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        """
        Initialize the hidden state to zeros.
        """
        return torch.zeros(1, self.hidden_size)
