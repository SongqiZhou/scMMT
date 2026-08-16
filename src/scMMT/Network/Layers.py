"""Neural-network building blocks used by scMMT."""

from torch.nn import SELU, BatchNorm1d, Dropout, Linear, Module


class Input_Block(Module):
    def __init__(self, in_units, out_units, dropout_inrate, dropout_outrate):
        super().__init__()
        self.bnorm_in = BatchNorm1d(in_units)
        self.dropout_in = Dropout(dropout_inrate)
        self.dense = Linear(in_units, out_units)
        self.dropout_out = Dropout(dropout_outrate)
        self.act = SELU()

    def forward(self, values):
        values = self.bnorm_in(values)
        values = self.dropout_in(values)
        values = self.dense(values)
        return self.dropout_out(self.act(values))


class Resnet(Module):
    def __init__(self, hidden_units, dropout_rate=0.1):
        super().__init__()
        self.dropout = Dropout(dropout_rate)
        self.dense = Linear(hidden_units, hidden_units)
        self.act = SELU()

    def forward(self, values):
        hidden = self.act(self.dense(self.dropout(values)))
        return values + hidden


class Resnet_last(Module):
    def __init__(self, hidden_units, dropout_rate=0.1):
        super().__init__()
        self.dropout = Dropout(dropout_rate)
        self.dense = Linear(hidden_units, hidden_units)
        self.act = SELU()

    def forward(self, values):
        return self.act(self.dense(self.dropout(values)))
