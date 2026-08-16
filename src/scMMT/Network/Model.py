"""Neural network and training loop for scMMT."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from anndata import AnnData
from pandas import concat
from torch import argmax, no_grad
from torch.nn import L1Loss, Linear, Module, Parameter, Sequential
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from .Layers import Input_Block, Resnet, Resnet_last


class scMMT_Model(Module):
    def __init__(
        self,
        p_mod1,
        p_mod2,
        loss1,
        loss2,
        categories,
        weight,
        h_size,
        drop_rate,
        n_layer,
        label_smoothing,
    ):
        super().__init__()
        if h_size <= 0:
            raise ValueError("h_size must be positive")
        if n_layer < 0:
            raise ValueError("n_layer cannot be negative")

        self.p_mod2 = int(p_mod2)
        self.h_size = int(h_size)
        self.input_block = Input_Block(p_mod1, h_size, drop_rate, drop_rate)
        self.resnet = Sequential(*[Resnet(h_size, dropout_rate=drop_rate) for _ in range(n_layer)])
        self.resnet_last = Resnet_last(h_size, dropout_rate=drop_rate)
        self.mod2_out = Linear(h_size, p_mod2)

        if categories is not None:
            self.celltype_out = Linear(h_size, len(categories))
            self.categories_arr = np.empty(len(categories), dtype=object)
            for category, index in categories.items():
                self.categories_arr[index] = category
        else:
            self.celltype_out = None
            self.categories_arr = None

        self.loss1 = loss1
        self.loss2 = loss2
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, x):
        x = x.to(torch.float32)
        hidden = self.resnet(self.input_block(x))
        embedding = self.resnet_last(hidden)
        return {
            "celltypes": (self.celltype_out(embedding) if self.celltype_out is not None else None),
            "modality 2": self.mod2_out(embedding),
            "embedding": embedding,
        }

    def _classification_loss(self, outputs, targets):
        return self.loss1(
            outputs["celltypes"],
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )

    def _validate(self, loader, device):
        if len(loader) == 0:
            raise ValueError("Validation loader is empty")

        self.eval()
        total_cells = 0
        correct = 0
        type_loss_sum = 0.0
        protein_loss_sum = 0.0

        with no_grad():
            for mod1, mod2, protein_bools, celltypes in loader:
                outputs = self(mod1)
                batch_size = len(mod1)
                total_cells += batch_size

                if self.categories_arr is not None:
                    batch_type_loss = self._classification_loss(outputs, celltypes)
                    type_loss_sum += batch_type_loss.item() * batch_size
                    correct += (argmax(outputs["celltypes"], dim=1) == celltypes).sum().item()

                if self.p_mod2:
                    batch_protein_loss = self.loss2(outputs["modality 2"], mod2, protein_bools)
                    protein_loss_sum += batch_protein_loss.item() * batch_size

        metrics = {
            "type_loss": type_loss_sum / total_cells if self.categories_arr is not None else 0.0,
            "protein_loss": protein_loss_sum / total_cells if self.p_mod2 else 0.0,
            "accuracy": correct / total_cells if self.categories_arr is not None else None,
        }
        metrics["selection_loss"] = metrics["type_loss"] + metrics["protein_loss"]
        return metrics

    def train_backprop(
        self,
        train_loader,
        val_loader,
        n_epoch=10000,
        ES_max=30,
        decay_max=10,
        decay_step=0.1,
        lr=1e-3,
        path=None,
        device="cpu",
    ):
        """Train the model and restore the best validation checkpoint.

        When both tasks are enabled, the implementation follows GradNorm and
        learns two positive, normalized task weights.  RNA-only references use
        ordinary classification training instead of attempting MSE over an
        empty protein matrix.
        """

        if n_epoch <= 0:
            raise ValueError("n_epoch must be positive")
        if lr <= 0 or decay_step <= 0:
            raise ValueError("lr and decay_step must be positive")
        if ES_max < 0 or decay_max <= 0:
            raise ValueError("ES_max must be non-negative and decay_max positive")
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("Training and validation loaders must not be empty")
        if self.categories_arr is None and self.p_mod2 == 0:
            raise ValueError("At least one prediction task must be enabled")

        model_optimizer = Adam(self.parameters(), lr=lr)
        model_scheduler = StepLR(model_optimizer, step_size=1, gamma=decay_step)
        task_weights = None
        weight_optimizer = None
        weight_scheduler = None
        initial_losses = None
        gradnorm_alpha = 0.15

        if self.categories_arr is not None and self.p_mod2:
            task_weights = Parameter(torch.ones(2, device=device))
            weight_optimizer = Adam([task_weights], lr=lr)
            weight_scheduler = StepLR(weight_optimizer, step_size=1, gamma=decay_step)

        best_loss = float("inf")
        best_state = copy.deepcopy(self.state_dict())
        patience = 0
        history = []

        for epoch in range(n_epoch):
            self.train()
            for mod1, mod2, protein_bools, celltypes in train_loader:
                outputs = self(mod1)
                raw_losses = []
                if self.categories_arr is not None:
                    raw_losses.append(self._classification_loss(outputs, celltypes))
                if self.p_mod2:
                    raw_losses.append(self.loss2(outputs["modality 2"], mod2, protein_bools))

                model_optimizer.zero_grad()
                if task_weights is None:
                    raw_losses[0].backward()
                else:
                    losses = torch.stack(raw_losses)
                    if initial_losses is None:
                        initial_losses = losses.detach().clamp_min(1e-8)

                    shared_weight = self.input_block.dense.weight
                    gradient_norms = []
                    for index, task_loss in enumerate(losses):
                        gradient = torch.autograd.grad(
                            task_weights[index] * task_loss,
                            shared_weight,
                            retain_graph=True,
                            create_graph=True,
                        )[0]
                        gradient_norms.append(torch.linalg.vector_norm(gradient))
                    gradient_norms = torch.stack(gradient_norms)

                    relative_losses = losses.detach() / initial_losses
                    inverse_rates = relative_losses / relative_losses.mean()
                    targets = gradient_norms.detach().mean() * inverse_rates.pow(gradnorm_alpha)
                    grad_loss = L1Loss(reduction="sum")(gradient_norms, targets)
                    weight_gradient = torch.autograd.grad(
                        grad_loss, task_weights, retain_graph=True
                    )[0]

                    # Task weights must not alter gradients of the network model.
                    model_loss = torch.sum(task_weights.detach() * losses)
                    model_loss.backward()

                    weight_optimizer.zero_grad()
                    task_weights.grad = weight_gradient

                model_optimizer.step()
                if task_weights is not None:
                    weight_optimizer.step()
                    with no_grad():
                        task_weights.clamp_(min=1e-3)
                        task_weights.mul_(len(task_weights) / task_weights.sum())

            metrics = self._validate(val_loader, device)
            history.append(metrics)
            accuracy_text = (
                f", accuracy={metrics['accuracy']:.3f}" if metrics["accuracy"] is not None else ""
            )
            print(
                f"Epoch {epoch}: protein loss={metrics['protein_loss']:.3f}, "
                f"cell-type loss={metrics['type_loss']:.3f}{accuracy_text}"
            )

            if metrics["selection_loss"] < best_loss:
                best_loss = metrics["selection_loss"]
                best_state = copy.deepcopy(self.state_dict())
                patience = 0
            else:
                patience += 1

            if patience and patience % decay_max == 0:
                model_scheduler.step()
                if weight_scheduler is not None:
                    weight_scheduler.step()
                print(f"Decaying learning rate to {model_optimizer.param_groups[0]['lr']}")
            if patience > ES_max:
                break

        self.load_state_dict(best_state)
        if path is not None:
            checkpoint_path = Path(path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), checkpoint_path)
        return history

    def impute(self, impute_loader, proteins):
        imputed = proteins.copy()
        self.eval()
        start = 0
        with no_grad():
            for mod1, bools, _ in impute_loader:
                end = start + mod1.shape[0]
                predicted = self(mod1)["modality 2"]
                imputed.X[start:end] = self.fill_predicted(imputed.X[start:end], predicted, bools)
                start = end
        return imputed

    def embed(self, impute_loader, test_loader, cells_train, cells_test):
        n_test = 0 if cells_test is None else len(cells_test)
        embedding = AnnData(np.zeros((len(cells_train) + n_test, self.h_size), dtype=np.float32))
        embedding.obs = (
            concat((cells_train, cells_test), join="outer")
            if cells_test is not None
            else cells_train.copy()
        )

        self.eval()
        start = 0
        with no_grad():
            for mod1, _, _ in impute_loader:
                end = start + mod1.shape[0]
                embedding.X[start:end] = self(mod1)["embedding"].cpu().numpy()
                start = end
            if test_loader is not None:
                for mod1 in test_loader:
                    end = start + mod1.shape[0]
                    embedding.X[start:end] = self(mod1)["embedding"].cpu().numpy()
                    start = end
        if start != embedding.n_obs:
            raise RuntimeError(
                f"Embedding loaders produced {start} cells; expected {embedding.n_obs}"
            )
        return embedding

    @staticmethod
    def fill_predicted(array, predicted, bools):
        observed = bools.detach().cpu().numpy()
        return (1.0 - observed) * predicted.detach().cpu().numpy() + np.asarray(array)

    def predict(self, test_loader, proteins, cells):
        predicted_data = AnnData(np.zeros((len(cells), self.p_mod2), dtype=np.float32))
        predicted_data.obs = cells.copy()
        predicted_data.var.index = proteins.var.index.copy()
        celltypes = [None] * len(cells) if self.categories_arr is not None else None

        self.eval()
        start = 0
        with no_grad():
            for mod1 in test_loader:
                end = start + mod1.shape[0]
                outputs = self(mod1)
                if celltypes is not None:
                    predicted_types = argmax(outputs["celltypes"], dim=1).cpu().numpy()
                    celltypes[start:end] = self.categories_arr[predicted_types].tolist()
                predicted_data.X[start:end] = outputs["modality 2"].cpu().numpy()
                start = end

        if start != predicted_data.n_obs:
            raise RuntimeError(
                f"Prediction loader produced {start} cells; expected {predicted_data.n_obs}"
            )

        if celltypes is not None:
            predicted_data.obs["transferred cell labels"] = celltypes
            # Backward-compatible alias for notebooks created before v1.1.
            predicted_data.obs["transfered cell labels"] = celltypes
        return predicted_data
