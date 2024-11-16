"""Test the WormSimulationsSplitStrategy class."""

import itertools
import re

import numpy as np
from pytest_cases import case, parametrize_with_cases

from dmb.data.bose_hubbard_2d.worm.dataset import WormSimulationsSplitStrategy
from dmb.data.split import IdDatasetSplitStrategy, Split
from tests.data.fake_id_dataset import FakeIdDataset


@case(tags=("split_strategy", "worm"))
def case_worm_simulations_split_strategy() -> IdDatasetSplitStrategy:
    """Return an instance of the WormSimulationsSplitStrategy."""
    return WormSimulationsSplitStrategy()


@case(tags=("id_dataset", "worm"))
def case_id_dataset_worm() -> FakeIdDataset:
    """Return a fake id dataset."""
    return FakeIdDataset(
        ids=("a", "a_tune", "b_tune", "c_tune", "d", "e", "e_tune", "f", "g", "g_tune"),
        indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    )


class TestWormSimulationsSplitStrategy:
    """Test the Split class."""

    @staticmethod
    @parametrize_with_cases("split_strategy",
                            cases=[case_worm_simulations_split_strategy])
    @parametrize_with_cases("id_dataset", cases=[case_id_dataset_worm])
    def test_split_worm_id_dataset(id_dataset: FakeIdDataset,
                                   split_strategy: IdDatasetSplitStrategy) -> None:
        """Test worm dataset split."""

        for _ in range(100):
            split_abs_sizes = {
                f"split_fraction_{idx}": int(np.random.randint(1, 100))
                for idx in range(np.random.randint(1, 5))
            }
            split_fractions = {
                key: value / sum(split_abs_sizes.values())
                for key, value in split_abs_sizes.items()
            }

            split_ids = Split.generate(
                split_fractions=split_fractions,
                dataset=id_dataset,
                split_strategy=split_strategy,
            )

            assert sum(len(ids) for ids in split_ids.values()) == len(id_dataset)
            simulation_ids = {
                key: set(re.sub("_tune", "", sample_id) for sample_id in value)
                for key, value in split_ids.items()
            }

            # check pairwise intersection are empty
            for key1, key2 in itertools.combinations(simulation_ids, 2):
                assert len(simulation_ids[key1].intersection(simulation_ids[key2])) == 0
