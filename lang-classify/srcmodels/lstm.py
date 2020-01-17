"""
LSTM version of the model.
"""
import torch


class LSTMModel(torch.nn.LSTM):
    def __init__(self, *args, **kwargs):
        """
        LSTM for classification of languages.
        """
        super().__init__(*args, **kwargs)
