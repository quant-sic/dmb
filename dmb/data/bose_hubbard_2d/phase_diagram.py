import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dmb.data.bose_hubbard_2d.network_input import net_input_dimless_const_parameters


def phase_diagram_uniform_inputs_iter(
    n_samples, zVU=1.0, muU_range=(-0.1, 3.1), ztU_range=(0.05, 0.85)
):
    muU = np.linspace(*muU_range, n_samples)
    ztU = np.linspace(*ztU_range, n_samples)

    MUU, ZTU = np.meshgrid(muU, ztU)

    MUU = MUU.flatten()
    ZTU = ZTU.flatten()

    for i in range(n_samples * n_samples):
        yield MUU[i], ZTU[i], net_input_dimless_const_parameters(
            muU=np.full((16, 16), fill_value=MUU[i]),
            ztU=ZTU[i],
            zVU=zVU,
            cb_projection=True,
        )


def phase_diagram_uniform_inputs(n_samples, zVU=1.0):
    MUU, ZTU, inputs = zip(*list(phase_diagram_uniform_inputs_iter(n_samples, zVU=zVU)))

    MUU = torch.from_numpy(np.array(MUU)).float()
    ZTU = torch.from_numpy(np.array(ZTU)).float()
    inputs = torch.stack(inputs, dim=0)

    return MUU, ZTU, inputs


def model_predict(model, inputs, batch_size=128):
    dl = DataLoader(inputs, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        outputs = []
        for inputs in tqdm(dl, desc="Predicting"):
            inputs = inputs.to(model.device)
            outputs.append(model(inputs))

        outputs = torch.cat(outputs, dim=0).to("cpu").detach()

    return outputs


def plot_phase_diagram(model, n_samples=250, zVU=1.0):
    MUU, ZTU, inputs = phase_diagram_uniform_inputs(n_samples=n_samples, zVU=zVU)
    outputs = model_predict(model, inputs, batch_size=512)

    density = outputs[:, model.observables.index("Density_Distribution")]

    reductions = {
        "mean": lambda x: x.mean(dim=(-1, -2)),
        "std": lambda x: x.std(dim=(-1, -2)),
        "max-min": lambda x: (x.amax(dim=(-1, -2)) - x.amin(dim=(-1, -2))),
        "max": lambda x: x.amax(dim=(-1, -2)),
        "min": lambda x: x.amin(dim=(-1, -2)),
    }

    figures_out = {}

    for name, reduction in reductions.items():
        figures_out[name] = plt.figure()

        plt.pcolormesh(
            ZTU.view(n_samples, n_samples).cpu().numpy(),
            MUU.view(n_samples, n_samples).cpu().numpy(),
            reduction(density).view(n_samples, n_samples).cpu().numpy(),
        )

        if zVU == 1.0:
            plt.ylim([0, 3])
            plt.xlim([0.1, 0.5])

            if name == "max-min":
                plt.clim([0, 1])

        elif zVU == 1.5:
            plt.ylim([0, 3])
            plt.xlim([0.1, 0.8])

            if name == "max-min":
                plt.clim([0, 3])
        else:
            raise RuntimeError(f"No plot specifications given for zVU={zVU}")

        plt.xlabel(r"$4J/U$")
        plt.ylabel(r"$\mu/{U}$")
        plt.colorbar()
        plt.tight_layout()

        plt.close()

    return figures_out
