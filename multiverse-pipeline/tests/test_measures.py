"""Unit tests for measure and metric primitives on hand-built inputs."""
import numpy as np
import pandas as pd
import pytest

from pipeline.measures import (consensus, delta, delta_composition, matched_pairs, neighbour_counts,
                               rank_z, smooth, stratified_sample)
from pipeline.metrics import graph_ilisi, kbet_like, majority_vote_disagreement, morans_i
from pipeline.integrate import FACTORS


def test_delta_identical_disjoint_partial():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(delta(A, A), 0)
    B = np.array([[7, 8, 9], [4, 5, 9]])
    d = delta(A, B)
    assert d[0] == pytest.approx(1.0)
    assert d[1] == pytest.approx(1 - 2 / 4)  # |∩|=2, |∪|=4


def test_delta_order_invariant():
    A = np.array([[3, 1, 2]])
    B = np.array([[2, 3, 1]])
    assert delta(A, B)[0] == pytest.approx(0)


def test_delta_composition():
    Ca = np.array([[3, 0, 0], [1, 1, 1]], np.uint8)
    Cb = np.array([[0, 3, 0], [1, 1, 1]], np.uint8)
    assert np.allclose(delta_composition(Ca, Cb), [1, 0])


def test_neighbour_counts_and_consensus():
    # 3 models, 4 cells, k=2; cell 0 always has neighbours {1,2}
    nbrs = np.array([
        [[1, 2], [0, 2], [0, 1], [0, 1]],
        [[1, 2], [0, 3], [0, 3], [0, 2]],
        [[2, 1], [2, 3], [1, 3], [1, 2]],
    ])
    row, ids, cnt = neighbour_counts(np.transpose(nbrs, (1, 0, 2)).reshape(4, -1), 4)
    assert dict(zip(ids[row == 0], cnt[row == 0])) == {1: 3, 2: 3}
    cons, stab = consensus(nbrs, 2, threshold=0.8)
    assert set(cons[0]) == {1, 2}
    assert stab[0] == pytest.approx(1.0)
    assert stab[3] < 1.0


def test_smooth_constant_is_constant():
    ref = np.array([[1, 2], [0, 2], [0, 1]])
    assert np.allclose(smooth(np.full(3, 0.4, np.float32), ref), 0.4)


def test_rank_z_monotone():
    pool = np.tile(np.linspace(0, 1, 21), (3, 1))
    z = rank_z(np.array([0.05, 0.5, 0.95]), pool)
    assert z[0] < z[1] < z[2]
    assert abs(z[1]) < 1e-6


def test_matched_pairs_full_factorial():
    import itertools
    levels = dict(n_hvg=[500, 1500], hvg_batch_aware=[False, True], exclude_igx=[False, True],
                  batch_key=["study", "sample"], method=["pca", "harmony", "combat_pca"])
    rows = [dict(zip(FACTORS, v)) for v in itertools.product(*(levels[f] for f in FACTORS))]
    cfgs = pd.DataFrame(rows)
    cfgs["config_id"] = [str(i) for i in range(len(cfgs))]
    pairs = matched_pairs(cfgs)
    counts = pairs.factor.value_counts().to_dict()
    assert counts == {"n_hvg": 24, "hvg_batch_aware": 24, "exclude_igx": 24, "batch_key": 24, "method": 48}
    # every pair differs in exactly its factor
    lut = cfgs.set_index("config_id")
    for r in pairs.itertuples():
        diff = [f for f in FACTORS if lut.loc[r.config_a, f] != lut.loc[r.config_b, f]]
        assert diff == [r.factor]


def test_stratified_sample_proportions():
    labels = np.array([0] * 800 + [1] * 200)
    idx = stratified_sample(labels, 100, seed=1)
    assert abs((labels[idx] == 1).mean() - 0.2) < 0.02
    assert len(np.unique(idx)) == len(idx)


def test_morans_i_signs():
    n, k = 200, 5
    nbrs = np.array([[(i + j) % n for j in range(1, k + 1)] for i in range(n)])
    smooth_x = np.sin(np.arange(n) / n * 2 * np.pi)
    rng = np.random.default_rng(0)
    assert morans_i(smooth_x, nbrs) > 0.8
    assert abs(morans_i(rng.normal(size=n), nbrs)) < 0.2


def test_batch_mixing_metrics():
    n, k = 300, 10
    batch = np.repeat([0, 1, 2], 100)
    label = np.zeros(n, int)
    rng = np.random.default_rng(0)
    mixed = rng.integers(0, n, (n, k))
    separated = np.array([rng.choice(np.flatnonzero(batch == batch[i]), k) for i in range(n)])
    assert graph_ilisi(mixed, batch) > 0.6 and graph_ilisi(separated, batch) < 0.05
    assert kbet_like(mixed, batch, label, 0.05) > 0.7 and kbet_like(separated, batch, label, 0.05) < 0.05


def test_majority_vote_disagreement():
    labels = np.array([0, 0, 1, 1])
    clusters = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    mv = majority_vote_disagreement(clusters, labels)
    assert mv[0] == 0 and mv[3] == 0
    assert mv[2] == pytest.approx(1 / 3)
