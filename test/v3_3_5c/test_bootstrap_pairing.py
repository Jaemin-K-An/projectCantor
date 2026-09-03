import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path("scripts/v3_3_5c").resolve()))
from analyse_generation import paired_max_t


def test_one_prompt_index_matrix_is_reused_for_all_rho_contrasts():
    rng = np.random.default_rng(1)
    differences = rng.normal(size=(25, 6))
    result = paired_max_t(differences, n_boot=100, seed=8)
    idx = result["indices"]
    assert idx.shape == (100, 25)
    for contrast in range(6):
        want = differences[:, contrast][idx].mean(axis=1)
        np.testing.assert_allclose(result["bootstrap"][:, contrast], want)


def test_pairing_is_preserved_under_prompt_offsets():
    base = np.arange(20, dtype=float)
    differences = np.column_stack([base, base + 100])
    result = paired_max_t(differences, n_boot=100, seed=2)
    np.testing.assert_allclose(result["bootstrap"][:, 1] - result["bootstrap"][:, 0], 100)
