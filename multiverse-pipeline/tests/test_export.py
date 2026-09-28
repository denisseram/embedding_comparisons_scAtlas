"""Schema tests for the web export, plus leaderboard sanity (validation check V7)."""
import json

import numpy as np
import pandas as pd
import pytest

from pipeline.common import load_config, out_dir, p

cfg = load_config()
EX = p(cfg, "export_dir")
pytestmark = pytest.mark.skipif(not (EX / "manifest.json").exists(), reason="run `make export` first")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((EX / "manifest.json").read_text())


@pytest.fixture(scope="module")
def models():
    return json.loads((EX / "models.json").read_text())


def test_files_exist_with_listed_sizes(manifest):
    for name, f in manifest["files"].items():
        path = EX / name
        assert path.exists(), name
        if name != "manifest.json":
            assert path.stat().st_size == f["bytes"], name


def test_binary_shapes_match_bytes(manifest):
    for name, f in manifest["files"].items():
        if f["dtype"] in ("json",):
            continue
        assert np.prod(f["shape"]) * np.dtype(f["dtype"]).itemsize == f["bytes"], name


def test_shapes_consistent_with_manifest(manifest):
    d = manifest["dataset"]
    N, M, P = d["n_cells"], d["n_models"], d["n_pairs"]
    files = manifest["files"]
    assert files["agreement.bin"]["shape"] == [M, M]
    assert files["pair_delta.bin"]["shape"] == [N, P]
    assert files["z_vs_ref.bin"]["shape"] == [len(manifest["references"]), N, M]
    assert files["cell_measures.bin"]["shape"] == [len(manifest["cell_measures"]), N]
    assert files["neighbors_sample.bin"]["shape"][:2] == [len(manifest["neighbor_models"]), N]
    assert len(manifest["model_order"]) == M and len(manifest["pairs"]) == P


def test_umap_models(manifest):
    if not manifest.get("umap_models"):
        pytest.skip("per-model UMAPs disabled")
    d = manifest["dataset"]
    assert manifest["files"]["umap_models.bin"]["shape"] == [d["n_models"], d["n_cells"], 2]
    lo, hi = np.array(manifest["umap_models"]["min"]), np.array(manifest["umap_models"]["max"])
    assert lo.shape == hi.shape == (d["n_models"], 2) and (hi >= lo).all()


def test_agreement_symmetric_zero_diagonal(manifest):
    M = manifest["dataset"]["n_models"]
    A = np.fromfile(EX / "agreement.bin", "<f4").reshape(M, M)
    assert np.allclose(A, A.T) and np.allclose(np.diag(A), 0)
    assert (A >= 0).all() and (A <= 1).all()


def test_cells_json(manifest):
    cells = json.loads((EX / "cells.json").read_text())
    N = manifest["dataset"]["n_cells"]
    assert len(cells["id"]) == len(cells["x"]) == len(cells["y"]) == N
    for c in cells["categorical"].values():
        assert len(c["codes"]) == N and max(c["codes"]) < len(c["levels"])


def test_models_json_order(manifest, models):
    assert [m["model_id"] for m in models] == manifest["model_order"]


def test_budget(manifest):
    total = sum(f.stat().st_size for f in EX.iterdir())
    assert total <= cfg["export"]["budget_mb"] * 1e6


# ---- V7 leaderboard sanity ---------------------------------------------------------------
def test_scaled_metrics_in_unit_interval(manifest, models):
    for m in models:
        for v in list(m["scaled"].values()) + [m["overall"], m["bio"], m["batch"]]:
            assert 0 - 1e-6 <= v <= 1 + 1e-6


def test_top10_matches_pipeline_metrics(manifest, models):
    met = pd.read_csv(out_dir(cfg) / "metrics.csv", index_col=0)
    for name in [x["name"] for x in manifest["metrics"]] + ["overall", "bio", "batch"]:
        col = name
        exported = sorted(models, key=lambda m: -(m[name] if name in ("overall", "bio", "batch") else m["raw"][name]))
        top_export = [m["model_id"] for m in exported[:10]]
        top_pipeline = met[col].sort_values(ascending=False, kind="stable").index[:10].tolist()
        # ties can reorder; compare the value at each rank
        v_exp = [met.loc[i, col] for i in top_export]
        v_pip = [met.loc[i, col] for i in top_pipeline]
        assert np.allclose(v_exp, v_pip, atol=1e-4), name


def test_seed_sd_present(manifest, models):
    for m in models:
        for name in [x["name"] for x in manifest["metrics"]] + ["overall", "bio", "batch"]:
            assert m["seed_sd"][name] is not None and m["seed_sd"][name] >= 0
