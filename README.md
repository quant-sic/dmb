# Many Body Density Prediction

Code for the paper [Deep learning of spatial densities in inhomogeneous correlated quantum systems](https://arxiv.org/pdf/2211.09050)

## Setup

This project is build with [UV](https://uv.readthedocs.io/en/latest/) and [Taskfile](https://taskfile.dev/). 

- Install [Task](https://taskfile.dev/installation/)
  (Add autocompletion for ease of use.)
```bash
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin v3.42.1
```

- Install [UV](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)
```bash
curl -LsSf https://astral.sh/uv/0.7.14/install.sh | sh
```

To install the dependencies, run:

```bash
task install
```

## Bose Hubbard Model in 2D

<p align="center">
    <img src="./docs/box_cuts.png" alt="Box cuts" width="49%"/>
    <img src="./docs/inversion.png" alt="Inversion" width="49%"/>
</p>

### Data Generation

The Bose Hubbard model in 2D is simulated with [QMC code](https://github.com/quant-sic/worm) adapted from [here](https://github.com/LodePollet/worm) to handle non-uniform chemical potentials.

After installation, random potentials can be simulated using:

```bash
uv run --no-sync python src/dmb/scripts/data/bose_hubbard_2d/worm/random_potential.py --potential-type random --number-of-samples <number_of_samples> --number-of-concurrent-jobs 1 --max-density-error <max_density_error>
```

A training dataset can be generated with:

```bash
uv run --no-sync python src/dmb/scripts/data/bose_hubbard_2d/worm/load_dataset.py <path_to_your_simulations> <target_dataset_path>
```

### Training

Model training is set up with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). The training script is `train.py`.

The model and training is configured with [Hydra](https://hydra.cc/).

#### Running an Experiment

To define an experiment, create a new configuration file in the `src/dmb/scripts/train/configs/experiment` directory. The configuration file should define the model, dataset, and training parameters. Default values are defined in `src/dmb/scripts/train/configs/**/*.yaml`.

To start training, run the following command:

```bash
uv run --no-sync python src/dmb/scripts/train/train.py experiment=path/to/your/config.yaml
```

### Inversion

Add an inversion configuration file in `src/dmb/scripts/invert/configs/experiment`. The configuration file should define the model, dataset, and inversion parameters.

To run the inversion, use the following command:

```bash
uv run --no-sync python src/dmb/scripts/invert/invert.py experiment=path/to/your/config.yaml
```

To simulate the inverted potential with QMC, use the following command:

```bash
uv run --no-sync python src/dmb/scripts/data/bose_hubbard_2d/worm/from_potential.py <path_to_your_inverted_potential> --number-of-concurrent-jobs 1 --max-density-error <max_density_error>
```
