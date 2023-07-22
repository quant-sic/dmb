from venv import create
import torch
from torch import nn
import pytorch_lightning as pl
from dmb.model.torch.loss import WeightedMAE,IndexMSELoss
from .simple_resnet1d import ResNet1d
from .simple_resnet2d import ResNet2d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Union, Literal
from dmb.misc import create_logger

logger = create_logger(__name__)


def combined_loss_criterion(loss, val_loss, alpha=1):
    return val_loss + abs(val_loss-loss)**alpha


class Model1d(nn.Module):
    def __init__(
        self,
        number_of_filters,
        kernel_size_first=5,
        kernel_size_rest=3,
        depth=19,
        activation="relu",
        nr_of_observables=4,
        use_batch_norm=False
    ):

        layers = []

        if activation == "relu":
            activation_func = nn.ReLU
        else:
            raise ValueError(f"Activation {activation} not handled!")

        # first layer
        layers.append(
            nn.Conv1d(
                in_channels=1,
                out_channels=number_of_filters,
                kernel_size=kernel_size_first,
                padding=kernel_size_first // 2,  # for dilation 1
                padding_mode="circular",
                bias=not use_batch_norm
            )
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(number_of_filters))

        layers.append(activation_func())

        for _ in range(1, depth - 1):
            layers.append(
                nn.Conv1d(
                    in_channels=number_of_filters,
                    out_channels=number_of_filters,
                    kernel_size=kernel_size_rest,
                    padding=kernel_size_rest // 2,  # for dilation 1
                    padding_mode="circular",
                    bias=not use_batch_norm
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(number_of_filters))
            layers.append(activation_func())

        # end of NN
        layers.append(
            nn.Conv1d(
                in_channels=number_of_filters,
                out_channels=60,
                kernel_size=kernel_size_rest,
                padding=kernel_size_rest // 2,  # for dilation 1
                padding_mode="circular",
                bias=not use_batch_norm
            )
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(60))

        layers.append(nn.Tanh())

        layers.append(
            nn.Conv1d(
                in_channels=60,
                out_channels=nr_of_observables,
                kernel_size=kernel_size_rest,
                padding=kernel_size_rest // 2,  # for dilation 1
                padding_mode="circular",
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        if isinstance(x, (tuple, list)):
            out = tuple(self.net(_x) for _x in x)
        else:
            out = self.net(x)

        return out

    def set_keras_weights(self, weights):
        for (name, p), keras_weight in zip(self.named_parameters(), weights):

            if "bias" in name:
                p.data = torch.from_numpy(keras_weight)
            elif "weight" in name:
                p.data = torch.from_numpy(keras_weight).transpose(2, 0)


class LitModel1d(pl.LightningModule):

    def __init__(self,
                 number_of_filters,
                 loss_weighs,
                 kernel_size_first=5,
                 kernel_size_rest=3,
                 depth=19,
                 activation="relu",
                 nr_of_observables=4,
                 learning_rate=1e-5,
                 use_batch_norm=False,
                 net_type="base_cnn",
                 lr_scheduler: Union[None, Literal["reduce_on_plateau"]] = None, loss_type="mae") -> None:
        super().__init__()

        self.save_hyperparameters()

        if net_type == "base_cnn":
            self.model = Model1d(number_of_filters=number_of_filters,
                                 kernel_size_first=kernel_size_first,
                                 kernel_size_rest=kernel_size_rest,
                                 depth=depth,
                                 activation=activation,
                                 nr_of_observables=nr_of_observables,
                                 use_batch_norm=use_batch_norm)

        elif net_type == "resnet":
            self.model = ResNet1d(kernel_size_first=kernel_size_first, depth=depth, kernel_size_rest=kernel_size_rest,
                                  number_of_filters=number_of_filters, nr_of_observables=nr_of_observables, n_base_channels=32)
        else:
            raise ValueError(f"Unknown model type {net_type}")

        observable_mapping = {"density1": 0, "correlation": 1, "covariance": 2, 'r_value': 3, 'r_value_oscillation': 3, "density2": 4,
                              "density3": 5, "density_middle": 6, "current_correlation": 7, "current_correlation_excited_middle": 8, "density_matrix_off_diag": 9}

        mae_names = ['density1',
                     'correlation',
                     'covariance',
                     'r_value_oscillation',
                     'density2',
                     'density3',
                     'density_middle',
                     'current_correlation',
                     'current_correlation_excited_middle',
                     'density_matrix_off_diag']
        ame_names = ["r_value"]

        if loss_type=="mae":
            self.loss = WeightedMAE(weights=loss_weighs, mae_names=mae_names,
                                    ame_names=ame_names, observable_mapping=observable_mapping)
        elif loss_type == "mse":
            self.loss = IndexMSELoss(weights=loss_weighs, mae_names=mae_names,
                                    ame_names=ame_names, observable_mapping=observable_mapping)
        else:
            raise ValueError("Unknown loss type")

    def configure_optimizers(self):

        out_dict = {}
        adam_opt = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["learning_rate"])
        out_dict["optimizer"] = adam_opt

        if self.hparams["lr_scheduler"] == "reduce_on_plateau":
            lr_scheduler = {"scheduler": ReduceLROnPlateau(
                adam_opt, factor=.5, patience=5000), "monitor": "loss"}
            out_dict["lr_scheduler"] = lr_scheduler

        return out_dict

    def _shared_eval_step(self, batch, batch_idx):

        batch_in, labels = batch
        prediction = self.model(batch_in)

        batch_size = sum(len(l) for l in labels)
        loss_mean = (sum(self.loss(prediction=pred, label=label).sum()
                     for pred, label in zip(prediction, labels))/batch_size).mean()

        return loss_mean, batch_size

    def training_step(self, batch, batch_idx):

        loss_mean, batch_size = self._shared_eval_step(batch, batch_idx)
        self.log("loss", loss_mean.detach().item(), on_step=True,
                 on_epoch=True, batch_size=batch_size)

        return {"loss": loss_mean}

    def training_epoch_end(self, training_step_outputs):
        self.current_train_loss = torch.stack(
            list(map(lambda o: o["loss"], training_step_outputs))).mean().item()

    def validation_step(self, batch, batch_idx):

        loss_mean, batch_size = self._shared_eval_step(batch, batch_idx)

        self.log("val_loss", loss_mean.detach().item(), batch_size=batch_size)


class Model2d(nn.Module):
    def __init__(
        self,
        number_of_filters,
        kernel_size_first=5,
        kernel_size_rest=3,
        depth=19,
        use_batch_norm=False,
        in_channels=1
    ):
        super().__init__()

        layers = []
        # first layer
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=number_of_filters,
                kernel_size=kernel_size_first,
                padding=kernel_size_first // 2,  # for dilation 1
                padding_mode="circular",
                bias=not use_batch_norm
            )
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(number_of_filters))

        layers.append(nn.ReLU())

        for _ in range(1, depth - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=number_of_filters,
                    out_channels=number_of_filters,
                    kernel_size=kernel_size_rest,
                    padding=kernel_size_rest // 2,  # for dilation 1
                    padding_mode="circular",
                    bias=not use_batch_norm
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(number_of_filters))
            layers.append(nn.ReLU())

        # end of NN
        layers.append(
            nn.Conv2d(
                in_channels=number_of_filters,
                out_channels=60,
                kernel_size=kernel_size_rest,
                padding=kernel_size_rest // 2,  # for dilation 1
                padding_mode="circular",
                bias=not use_batch_norm
            )
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(60))

        layers.append(nn.Tanh())

        layers.append(
            nn.Conv2d(
                in_channels=60,
                out_channels=1,
                kernel_size=kernel_size_rest,
                padding=kernel_size_rest // 2,  # for dilation 1
                padding_mode="circular",
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        if isinstance(x, (tuple, list)):
            out = tuple(self.net(_x) for _x in x)
        else:
            out = self.net(x)

        return out

    def set_keras_weights(self, weights):
        for (name, p), keras_weight in zip(self.named_parameters(), weights):

            if "bias" in name:
                assert (torch.from_numpy(keras_weight).shape == p.shape)
                p.data = torch.from_numpy(keras_weight)
            elif "weight" in name:
                assert (torch.from_numpy(keras_weight).permute(
                    3, 2, 1, 0).shape == p.shape)
                p.data = torch.from_numpy(keras_weight).permute(3, 2, 1, 0)


class LitModel2d(pl.LightningModule):

    def __init__(self,
                 number_of_filters,
                 kernel_size_first=5,
                 kernel_size_rest=3,
                 depth=19,
                 activation="relu",
                 nr_of_observables=4,
                 learning_rate=1e-5,
                 use_batch_norm=False,
                 net_type="base_cnn",
                 in_channels=1,
                 dropout=0,
                 lr_scheduler: Union[None, Literal["reduce_on_plateau"]] = None) -> None:
        super().__init__()

        self.save_hyperparameters()

        if net_type == "base_cnn":
            self.model = Model2d(number_of_filters=number_of_filters,
                                 kernel_size_first=kernel_size_first,
                                 kernel_size_rest=kernel_size_rest,
                                 depth=depth,
                                 in_channels=in_channels,
                                 use_batch_norm=use_batch_norm)

        elif net_type == "resnet":
            self.model = ResNet2d(kernel_size_first=kernel_size_first, depth=depth, kernel_size_rest=kernel_size_rest,
                                  number_of_filters=number_of_filters, n_base_channels=32, in_channels=in_channels, dropout=dropout)
        else:
            raise ValueError(f"Unknown model type {net_type}")

        self.loss = nn.MSELoss()

    @property
    def current_train_loss(self):
        if hasattr(self, "_current_train_loss"):
            return self._current_train_loss
        else:
            self._current_train_loss = torch.inf
            return self._current_train_loss

    @current_train_loss.setter
    def current_train_loss(self, v):
        self._current_train_loss = v

    def configure_optimizers(self):

        out_dict = {}
        adam_opt = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["learning_rate"])
        out_dict["optimizer"] = adam_opt

        if self.hparams["lr_scheduler"] == "reduce_on_plateau":
            lr_scheduler = {"scheduler": ReduceLROnPlateau(
                adam_opt, factor=.5, patience=10000), "monitor": "loss"}
            out_dict["lr_scheduler"] = lr_scheduler

        return out_dict

    def _shared_eval_step(self, batch, batch_idx):

        batch_in, labels = batch
        prediction = self.model(batch_in)

        batch_size = sum(len(l) for l in labels)
        loss_mean = sum(self.loss(pred.squeeze(), label).sum()
                        for pred, label in zip(prediction, labels))/batch_size

        return loss_mean, batch_size

    def training_step(self, batch, batch_idx):

        loss_mean, batch_size = self._shared_eval_step(batch, batch_idx)
        self.log("loss", loss_mean.detach().item(), on_step=True,
                 on_epoch=True, batch_size=batch_size)

        return {"loss": loss_mean}

    def training_epoch_end(self, training_step_outputs):
        self.current_train_loss = torch.stack(
            list(map(lambda o: o["loss"], training_step_outputs))).mean().item()

    def validation_step(self, batch, batch_idx):

        loss_mean, batch_size = self._shared_eval_step(batch, batch_idx)

        for alpha in (.8, 1.0, 1.2, 1.4, 1.6):
            self.log(f"loss_crit/{alpha}", combined_loss_criterion(loss_mean.detach(
            ).item(), self.current_train_loss, alpha=alpha), batch_size=batch_size)

        self.log("val_loss", loss_mean.detach().item(), batch_size=batch_size)

    def test_step(self, batch, batch_idx):

        loss_mean, batch_size = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss_mean}

        self.log_dict(metrics, batch_size=batch_size)

        return metrics
