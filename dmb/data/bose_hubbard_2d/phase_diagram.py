from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dmb.data.bose_hubbard_2d.cpp_worm.dataset import BoseHubbardDataset
from dmb.data.bose_hubbard_2d.network_input import \
    net_input_dimless_const_parameters
from dmb.paths import REPO_DATA_ROOT


def phase_diagram_uniform_inputs_iter(n_samples,
                                      zVU=1.0,
                                      muU_range=(-0.1, 3.1),
                                      ztU_range=(0.05, 0.85)):
    muU = np.linspace(*muU_range, n_samples)
    ztU = np.linspace(*ztU_range, n_samples)

    MUU, ZTU = np.meshgrid(muU, ztU)

    MUU = MUU.flatten()
    ZTU = ZTU.flatten()

    # cb version 1
    fake_target_density = np.zeros((16, 16))
    fake_target_density[::2, ::2] = 1.0

    for i in range(n_samples * n_samples):
        yield MUU[i], ZTU[i], net_input_dimless_const_parameters(
            muU=np.full((16, 16), fill_value=MUU[i]),
            ztU=ZTU[i],
            zVU=zVU,
            cb_projection=True,
            target_density=fake_target_density,
        )


def phase_diagram_uniform_inputs(n_samples, zVU=1.0):
    MUU, ZTU, inputs = zip(
        *list(phase_diagram_uniform_inputs_iter(n_samples, zVU=zVU)))

    MUU = torch.from_numpy(np.array(MUU)).float()
    ZTU = torch.from_numpy(np.array(ZTU)).float()
    inputs = torch.stack(inputs, dim=0)

    return MUU, ZTU, inputs


def model_predict(model, inputs, batch_size=512):
    dl = DataLoader(inputs,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0)

    model.eval()
    with torch.no_grad():
        outputs = []
        for inputs in tqdm(dl, desc="Predicting", disable=True):
            inputs = inputs.to(model.device)
            outputs.append(model(inputs))

        outputs = torch.cat(outputs, dim=0).to("cpu").detach()

    return outputs


import numpy as np


def add_phase_boundaries(ax):
    red = [
        (0, 0),
        (2, 62),
        (3, 124),
        (3, 183),
        (3, 245),
        (3.5, 306),
        (4, 368),
        (5, 430),
        (6, 478),
        (6.5, 491),
        (7, 500),
        (8, 506),
        (8, 502),
        (9, 494),
        (10, 484),
        (13.5, 439),
        (18, 386),
        (24, 334),
        (32, 287),
        (42, 247),
        (55, 214),
        (72, 198),
        (90, 199),
        (73, 216),
        (60, 249),
        (51, 295),
        (45, 345),
        (42, 381),
        (42, 394),
        (42, 408),
        (43, 415),
        (45, 415),
        (47, 413),
        (53, 402),
        (62, 389),
        (71, 383),
        (81, 387),
        (90, 398),
        (81, 416),
        (74.5, 442),
        (71, 462),
        (68, 483),
        (65, 508),
        (64, 522),
        (64, 529),
        (65, 542),
        (66, 544),
        (67, 548),
        (71, 556),
        (74, 561),
        (78, 567),
        (84, 580),
        (90, 597),
    ]
    distances = (199, 230 / 0.15)

    red_dots = []
    for point in red:
        x, y = (
            np.cos(point[0] * np.pi / 180) * point[1] / distances[1],
            np.sin(point[0] * np.pi / 180) * point[1] / distances[0],
        )
        red_dots.append((x, y))

    ax.plot(*zip(*red_dots), marker="o", c="red")

    blue_1 = [
        (90, 199),
        (73, 210),
        (58, 235),
        (47, 271),
        (38, 313),
        (31, 358),
        (25, 403),
        (18.5, 450),
        (12.5, 499),
        (11, 509),
        (10, 513),
        (8, 515),
        (7, 508),
        (6.5, 503),
    ]
    blue_dots_1 = []
    for point in blue_1:
        x, y = (
            np.cos(point[0] * np.pi / 180) * point[1] / distances[1],
            np.sin(point[0] * np.pi / 180) * point[1] / distances[0],
        )
        blue_dots_1.append((x, y))

    ax.plot(*zip(*blue_dots_1), marker="o", c="blue")

    blue_2 = [
        (90, 398),
        (81, 406),
        (73, 425),
        (66, 453),
        (61, 495),
        (59, 527),
        (58, 555),
        (58, 568),
        (59, 581),
        (60, 590),
        (63, 605),
        (67, 616),
        (73, 618),
        (78, 610),
        (84, 602),
        (90, 597),
    ]
    blue_dots_2 = []
    for point in blue_2:
        x, y = (
            np.cos(point[0] * np.pi / 180) * point[1] / distances[1],
            np.sin(point[0] * np.pi / 180) * point[1] / distances[0],
        )
        blue_dots_2.append((x, y))

    ax.plot(*zip(*blue_dots_2), marker="o", c="blue")


def plot_phase_diagram(model, n_samples=250, zVU=1.0):
    MUU, ZTU, inputs = phase_diagram_uniform_inputs(n_samples=n_samples,
                                                    zVU=zVU)
    outputs = model_predict(model, inputs, batch_size=512)

    reductions = {
        "mean": lambda x: x.mean(dim=(-1, -2)),
        "std": lambda x: x.std(dim=(-1, -2)),
        "max-min": lambda x: (x.amax(dim=(-1, -2)) - x.amin(dim=(-1, -2))),
        "max": lambda x: x.amax(dim=(-1, -2)),
        "min": lambda x: x.amin(dim=(-1, -2)),
    }

    figures_out = defaultdict(dict)

    for obs in model.observables:
        output_obs = outputs[:, model.observables.index(obs)]

        for name, reduction in reductions.items():
            figures_out[obs][name] = plt.figure()

            plt.pcolormesh(
                ZTU.view(n_samples, n_samples).cpu().numpy(),
                MUU.view(n_samples, n_samples).cpu().numpy(),
                reduction(output_obs).view(n_samples, n_samples).cpu().numpy(),
            )

            if zVU == 1.0:
                add_phase_boundaries(plt.gca())
                plt.ylim([0, 3])
                plt.xlim([0.1, 0.5])

                if name == "max-min" and obs == "density":
                    plt.clim([0, 1])

            elif zVU == 1.5:
                plt.ylim([0, 3])
                plt.xlim([0.1, 0.8])

                if name == "max-min" and obs == "density":
                    plt.clim([0, 3])

            plt.xlabel(r"$4J/U$")
            plt.ylabel(r"$\mu/{U}$")
            plt.colorbar()
            plt.tight_layout()

            plt.close()

    return figures_out


def plot_phase_diagram_mu_cut(
    model,
    zVU: float = 1.0,
    ztU: float = 0.25,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 16,
    muU_num_steps: int = 50,
):
    path = REPO_DATA_ROOT / "mu_cut" / f"{zVU}/{ztU}/{L}"

    ds = BoseHubbardDataset(
        data_dir=path,
        clean=True,
        max_density_error=0.015,
        reload=True,
        verbose=False,
    )

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    inputs = torch.stack(
        [
            net_input_dimless_const_parameters(
                muU=np.full((L, L), fill_value=_muU),
                ztU=ztU,
                zVU=zVU,
                cb_projection=True,
                target_density=np.ones((L, L)),
            ) for _muU in muU
        ],
        dim=0,
    )
    outputs = model_predict(model, inputs, batch_size=512)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    try:
        muU_qmc, n_qmc = zip(*[(ds.phase_diagram_position(i)[1], ds_i[1][0])
                               for i, ds_i in enumerate(ds)])

        ax.scatter(muU_qmc, [n_qmc[i].max() for i in range(len(n_qmc))],
                   c="black",
                   label="QMC")
        ax.scatter(muU_qmc, [n_qmc[i].min() for i in range(len(n_qmc))],
                   c="black",
                   label="QMC")
    except:
        pass

    # max
    ax.scatter(
        muU,
        outputs.cpu()[:, 0].numpy().max(axis=(-1, -2)),
        c="red",
        label="NN",
    )

    ax.scatter(
        muU,
        outputs.cpu()[:, 0].numpy().min(axis=(-1, -2)),
        c="red",
        label="NN",
    )

    plt.legend()
    plt.xlabel(r"$\mu/U$")
    plt.ylabel(r"$n$")
    plt.tight_layout()
    plt.close()

    return {"mu_cut": fig}
