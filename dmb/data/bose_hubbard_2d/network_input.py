import math
from typing import Tuple, Union

import numpy as np
import torch


def get_ckeckerboard_projection(target_density: torch.Tensor):
    """
    Determines and returns the checkerboard version (out of two possible) that has the largest correlation with the input mu.
    """

    if len(target_density.shape) == 1:
        target_density = target_density.view(
            int(math.sqrt(target_density.shape[0])),
            int(math.sqrt(target_density.shape[0])),
        )
    elif len(target_density.shape) == 2:
        if not target_density.shape[0] == target_density.shape[1]:
            raise ValueError("Input mu has to be square")
    else:
        raise ValueError("Input mu has to be either 1D or 2D")

    cb_1 = torch.ones_like(target_density)
    cb_1[::2, ::2] = 0
    cb_1[1::2, 1::2] = 0

    cb_2 = torch.ones_like(target_density)
    cb_2[1::2, ::2] = 0
    cb_2[::2, 1::2] = 0

    corr_1 = torch.sum(target_density * cb_1)
    corr_2 = torch.sum(target_density * cb_2)

    if corr_1 > corr_2:
        return cb_1
    else:
        return cb_2


def net_input(
    mu: Union[torch.Tensor, np.ndarray],
    U_on: Union[torch.Tensor, np.ndarray, float],
    V_nn: Union[torch.Tensor, np.ndarray, float],
    cb_projection: bool = True,
    target_density: Union[torch.Tensor, np.ndarray] = None,
):
    # convert to torch.Tensor if necessary
    if isinstance(mu, np.ndarray):
        mu = torch.from_numpy(mu).float()

    if isinstance(U_on, np.ndarray):
        U_on = torch.from_numpy(U_on).float()
    elif isinstance(U_on, float):
        U_on = torch.full_like(mu, fill_value=U_on)

    if isinstance(V_nn, np.ndarray):
        V_nn = torch.from_numpy(V_nn).float()
    elif isinstance(V_nn, float):
        V_nn = torch.full_like(mu, fill_value=V_nn).expand(2, *mu.shape)

    # convert to 2D if necessary
    if len(mu.shape) == 1:
        mu = mu.view(int(math.sqrt(mu.shape[0])), int(math.sqrt(mu.shape[0])))
    elif len(mu.shape) == 2:
        if not mu.shape[0] == mu.shape[1]:
            raise ValueError("Input mu has to be square")
    else:
        raise ValueError("Input mu has to be either 1D or 2D")

    # convert to 2D if necessary
    if len(U_on.shape) == 1:
        U_on = U_on.view(int(math.sqrt(U_on.shape[0])), int(math.sqrt(U_on.shape[0])))
    elif len(U_on.shape) == 2:
        if not U_on.shape[0] == U_on.shape[1]:
            raise ValueError("Input U_on has to be square")
    else:
        raise ValueError("Input U_on has to be either 1D or 2D")

    # convert to 2D if necessary
    if len(V_nn.shape) == 1:
        V_nn = V_nn.view(
            2, int(math.sqrt(V_nn.shape[0] / 2)), int(math.sqrt(V_nn.shape[0] / 2))
        )
    elif len(V_nn.shape) == 3:
        if not V_nn.shape[1] == V_nn.shape[2]:
            raise ValueError("Input V_nn has to be square")
    else:
        raise ValueError("Input V_nn has to be either 1D or 3D")

    # get checkerboard projection
    if cb_projection:
        if target_density is None:
            raise RuntimeError(
                "If cb_projection is True, target_density has to be provided."
            )
        cb_proj = get_ckeckerboard_projection(target_density=target_density)
    else:
        cb_proj = torch.ones_like(target_density)

    # get network input
    inputs = torch.concat(
        [mu[None, :], cb_proj[None, :], U_on[None, :], V_nn[[0]]], dim=0
    )

    return inputs


def net_input_dimless_const_parameters(
    muU: Union[torch.Tensor, np.ndarray],
    ztU: float,
    zVU: float,
    cb_projection: bool = True,
    target_density: Union[torch.Tensor, np.ndarray] = None,
):
    # convert to torch.Tensor if necessary
    if isinstance(muU, np.ndarray):
        muU = torch.from_numpy(muU).float()

    # convert to 2D if necessary
    if len(muU.shape) == 1:
        muU = muU.view(int(math.sqrt(muU.shape[0])), int(math.sqrt(muU.shape[0])))
    elif len(muU.shape) == 2:
        if not muU.shape[0] == muU.shape[1]:
            raise ValueError("Input muU has to be square")
    else:
        raise ValueError("Input muU has to be either 1D or 2D")

    # get checkerboard projection
    if cb_projection:
        if target_density is None:
            raise RuntimeError(
                "If cb_projection is True, target_density has to be provided."
            )
        cb_proj = get_ckeckerboard_projection(target_density=target_density)
    else:
        cb_proj = torch.ones_like(target_density)

    # conversion
    U_on = torch.full(size=muU.shape, fill_value=4 / ztU)
    V_nn = zVU / 4 * U_on
    mu = muU * U_on

    # get network input
    inputs = torch.concat(
        [mu[None, :], cb_proj[None, :], U_on[None, :], V_nn[None, :]], dim=0
    )

    return inputs


def dimless_from_net_input(
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # split
    mu, cb, U_on, V_nn = torch.split(inputs, 1, dim=1)

    # conversion
    ztU = 4 / U_on
    zVU = 4 / U_on * V_nn
    muU = mu / U_on

    return muU, cb, ztU, zVU
