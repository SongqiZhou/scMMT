import pytest

from scMMT.Data_Infrastructure.Samplers import batchSampler, build_trainSamplers


class SizedData:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_batch_sampler_reports_batches_and_dataset_membership():
    sampler = batchSampler(range(7), train_keys=[3, 5], bsize=3)

    batches = list(sampler)

    assert len(sampler) == 3
    assert batches[0] == ([0, 1, 2], [0, 0, 0])
    assert batches[1] == ([3, 4, 5], [1, 1, 2])


def test_random_train_validation_split_is_complete_and_disjoint():
    train, validation = build_trainSamplers(
        SizedData(10), [], bsize=4, val_split=None, val_frac=0.2, seed=7
    )
    train_indices = {index for batch, _ in train for index in batch}
    validation_indices = {index for batch, _ in validation for index in batch}

    assert train_indices.isdisjoint(validation_indices)
    assert train_indices | validation_indices == set(range(10))
    assert len(validation_indices) == 2


def test_unresolved_by_test_split_has_actionable_error():
    with pytest.raises(ValueError, match="requires gene_test"):
        build_trainSamplers(SizedData(10), [], val_split="by_test", val_frac=0.2)
