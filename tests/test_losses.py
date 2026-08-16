import torch

from scMMT.Network.Losses import mse_loss


def test_masked_mse_averages_only_observed_entries():
    predicted = torch.tensor([[1.0, 10.0], [3.0, 20.0]])
    target = torch.zeros_like(predicted)
    observed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    assert torch.isclose(mse_loss()(predicted, target, observed), torch.tensor(5.0))


def test_masked_mse_is_differentiable_for_empty_panel():
    predicted = torch.ones((2, 0), requires_grad=True)
    target = torch.empty((2, 0))
    observed = torch.empty((2, 0))

    loss = mse_loss()(predicted, target, observed)
    loss.backward()

    assert loss.item() == 0.0
    assert predicted.grad is not None
