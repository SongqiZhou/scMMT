import numpy as np
import torch
from torch.nn.functional import cross_entropy

from scMMT.Network.Losses import mse_loss
from scMMT.Network.Model import scMMT_Model


def _batch(input_size=6, proteins=3):
    return [
        torch.randn(8, input_size),
        torch.randn(8, proteins),
        torch.ones(8, proteins),
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
    ]


def test_model_honours_configured_embedding_size():
    model = scMMT_Model(
        p_mod1=6,
        p_mod2=3,
        loss1=cross_entropy,
        loss2=mse_loss(),
        categories={"a": 0, "b": 1},
        weight=None,
        h_size=24,
        drop_rate=0.0,
        n_layer=2,
        label_smoothing=0.0,
    )

    outputs = model(torch.randn(4, 6))

    assert outputs["embedding"].shape == (4, 24)
    assert outputs["celltypes"].shape == (4, 2)
    assert outputs["modality 2"].shape == (4, 3)


def test_gradnorm_training_runs_and_writes_checkpoint(tmp_path):
    torch.manual_seed(4)
    model = scMMT_Model(
        p_mod1=6,
        p_mod2=3,
        loss1=cross_entropy,
        loss2=mse_loss(),
        categories={"a": 0, "b": 1},
        weight=None,
        h_size=12,
        drop_rate=0.0,
        n_layer=1,
        label_smoothing=0.0,
    )
    train_loader = [_batch()]
    validation_loader = [_batch()]
    checkpoint = tmp_path / "weights.pt"

    history = model.train_backprop(
        train_loader,
        validation_loader,
        n_epoch=1,
        ES_max=1,
        path=checkpoint,
    )

    assert checkpoint.is_file()
    assert len(history) == 1
    assert np.isfinite(history[0]["selection_loss"])
