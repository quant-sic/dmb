"""Tests for the BoseHubbard2dDataset."""

import json
from pathlib import Path

import torch
from pytest_cases import case, parametrize_with_cases

from dmb.data.bose_hubbard_2d.nn_input import convert_dimless_to_physical
from dmb.data.bose_hubbard_2d.worm.dataset import (
    BoseHubbard2dDataset,
    BoseHubbard2dSampleFilterStrategy,
)
from dmb.data.dataset import SampleFilterStrategy


class DatasetDirFilterCases:
    """Test cases for the BoseHubbard2dDataset."""

    @staticmethod
    @case(tags=("dataset_dir", "filter"))
    def case_dataset_dir_filter_sample_1(
        tmp_path: Path,
    ) -> tuple[Path, SampleFilterStrategy, list[str]]:
        """Return the dataset directory, filter strategy, and expected sample ids."""

        def convert_dimless(
            ztU: float = 0.5, muU: float = 1.0, zVU: float = 1.0, J: float = 1.0
        ) -> dict[str, float]:
            """Convert the parameters from dimensionless to physical units."""

            U_on, mu, V_nn = convert_dimless_to_physical(ztU, muU, zVU, J)

            return {
                "U_on": U_on,
                "mu": mu,
                "V_nn": V_nn,
            }

        base_metadata = {
            "max_density_error": 0.01,
            "J": 1.0,
            "L": 10,
            **convert_dimless(),
        }

        metadata = [
            base_metadata,  # ok
            {**base_metadata, **convert_dimless(muU=-0.4)},  # ok
            {**base_metadata, **convert_dimless(muU=-1.0)},  # bad
            {**base_metadata, **convert_dimless(muU=4.0)},  # bad
            {**base_metadata, **convert_dimless(ztU=0.025)},  # bad
            {**base_metadata, "L": 1},  # bad
            {**base_metadata, "max_density_error": 0.1},  # bad
            {**base_metadata, "L": 25, "max_density_error": 0.1},  # bad
        ]

        sample_ids = [f"sample_{idx}" for idx in range(len(metadata))]

        # save samples and metadata
        for idx, (meta, sample_id) in enumerate(zip(metadata, sample_ids)):
            sample_path = tmp_path / "samples" / sample_id
            sample_path.mkdir(parents=True, exist_ok=True)

            torch.save(torch.rand(10, 10), sample_path / "inputs.pt")
            torch.save(torch.rand(10, 10), sample_path / "outputs.pt")

            with open(sample_path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

        with open(tmp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({}, f)

        sample_filter_strategy = BoseHubbard2dSampleFilterStrategy(
            ztU_range=(0.05, 1.0),
            muU_range=(-0.5, 3.0),
            zVU_range=(0.75, 1.75),
            L_range=(2, 20),
            max_density_error=0.015,
        )

        return tmp_path, sample_filter_strategy, sample_ids[:2]


class TestBoseHubbard2dDataset:
    """Tests for the BoseHubbard2dDataset."""

    @staticmethod
    @parametrize_with_cases(
        "dataset_dir, sample_filter_strategy, expected_sample_ids",
        cases=DatasetDirFilterCases,
    )
    def test_dataset_dir_filter_sample(
        dataset_dir: Path,
        sample_filter_strategy: SampleFilterStrategy,
        expected_sample_ids: list[str],
    ) -> None:
        """Test loading the dataset directory with a filter strategy."""

        dataset = BoseHubbard2dDataset(
            dataset_dir_path=dataset_dir, sample_filter_strategy=sample_filter_strategy
        )

        assert len(dataset) == len(expected_sample_ids)
        assert set(dataset.sample_ids) == set(expected_sample_ids)
