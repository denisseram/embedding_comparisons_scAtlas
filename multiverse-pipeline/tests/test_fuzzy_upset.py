"""Fuzzy UpSet computation on hand-built graphs with known answers, plus the synthetic two-model case."""
import numpy as np
import pytest
import scipy.sparse as sp

from pipeline.fuzzy_upset import (DIFFUSE, NONE, aggregate, attributes, cells_for, compare, difference, label_codes,
                                  label_frequency, label_warnings, membership_counts, memberships,
                                  memberships_from_adjacency, ranked, signatures)

# 5 cells, labels a a b b <unlabelled>, k = 2 (self excluded; self-loop added -> rows of 3)
CODES = np.array([0, 0, 1, 1, -1], np.int32)
KNN = np.array([[1, 2], [0, 4], [3, 4], [0, 2], [2, 3]])
# neighbourhoods incl. self:  0:{0,1,2} 1:{1,0,4} 2:{2,3,4} 3:{3,0,2} 4:{4,2,3}
P_EXPECTED = np.array([[2, 1], [2, 0], [0, 2], [1, 2], [0, 2]]) / 3


def sig_of(s, i):
    return None if s.sig[i] < 0 else s.intersections[s.sig[i]]


def test_label_codes_unlabelled():
    codes, levels = label_codes(["x", None, "y", np.nan, "", "x"])
    assert levels == ["x", "y"]
    assert codes.tolist() == [0, -1, 1, -1, -1, 0]


def test_memberships_known_values():
    P = memberships(KNN, CODES, 2).toarray()
    assert np.allclose(P, P_EXPECTED)
    assert membership_counts(KNN, CODES, 2).dtype == np.uint8
    # unlabelled neighbours still count in the denominator: rows touching cell 4 sum to < 1
    assert np.allclose(P.sum(1), [1, 2 / 3, 2 / 3, 1, 2 / 3])


def test_memberships_chunked_equals_unchunked():
    rng = np.random.default_rng(0)
    knn = rng.integers(0, 1000, (1000, 7))
    codes = rng.integers(-1, 5, 1000).astype(np.int32)
    assert np.array_equal(membership_counts(knn, codes, 5, chunk=97), membership_counts(knn, codes, 5))


def test_adjacency_path_matches_knn_path():
    n = len(CODES)
    A = sp.csr_matrix((np.ones(KNN.size), (np.repeat(np.arange(n), 2), KNN.ravel())), shape=(n, n))
    assert np.allclose(memberships_from_adjacency(A, CODES, 2).toarray(), P_EXPECTED)
    # weighted input is binarised; an existing self-loop is not counted twice
    A2 = A * 5.0 + sp.identity(n)
    assert np.allclose(memberships_from_adjacency(A2, CODES, 2).toarray(), P_EXPECTED)


def test_signatures_and_strengths():
    s = signatures(memberships(KNN, CODES, 2), tau=0.1, max_size=3)
    assert [sig_of(s, i) for i in range(5)] == [(0, 1), (0,), (1,), (0, 1), (1,)]
    assert np.allclose(s.strength, [1 / 3, 2 / 3, 2 / 3, 1 / 3, 2 / 3])
    agg = aggregate(s).set_index("labels")
    assert agg.loc[[(0,)], "n_cells"].item() == 1 and agg.loc[[(1,)], "n_cells"].item() == 2
    assert agg.loc[[(0, 1)], "fuzzy_size"].item() == pytest.approx(2 / 3)
    assert agg.loc[[(1,)], "mean_strength"].item() == pytest.approx(2 / 3)
    assert cells_for(s, (1, 0)).tolist() == [0, 3]
    assert s.n_none == 0 and s.n_diffuse == 0


def test_diffuse_and_none_buckets():
    P = memberships(KNN, CODES, 2)
    s = signatures(P, tau=0.1, max_size=1)
    assert s.n_diffuse == 2 and s.sig[0] == DIFFUSE and s.sig[3] == DIFFUSE
    assert s.strength[0] == 0
    s = signatures(P, tau=0.5)
    assert [sig_of(s, i) for i in range(5)] == [(0,), (0,), (1,), (1,), (1,)]
    s = signatures(P, tau=0.7)
    assert s.n_none == 5 and (s.sig == NONE).all() and s.intersections == []


def test_normalization_thresholds_enrichment_but_strength_is_raw():
    P = memberships(KNN, CODES, 2)
    freq = label_frequency(CODES, 2)
    assert np.allclose(freq, [0.5, 0.5])  # among labelled cells
    s = signatures(P, tau=1.0, normalize=True, freq=freq)  # enrichment >= 1  <=>  P >= 0.5
    assert [sig_of(s, i) for i in range(5)] == [(0,), (0,), (1,), (1,), (1,)]
    assert np.allclose(s.strength, 2 / 3)
    with pytest.raises(ValueError):
        signatures(P, normalize=True)


def test_normalization_favours_rare_label():
    # cell 0 sees 1 rare (r) + 3 common (c): raw 0.25 fails tau=0.3, enrichment 0.25/0.1 passes
    codes = np.array([1] * 9 + [0], np.int32)  # label 0 = rare (1 of 10 cells)
    knn = np.array([[1, 2, 9]] + [[0, 1, 2]] * 9)
    P = memberships(knn, codes, 2)
    assert sig_of(signatures(P, tau=0.3), 0) == (1,)
    assert sig_of(signatures(P, tau=0.3, normalize=True, freq=label_frequency(codes, 2)), 0) == (0, 1)


def test_attributes():
    s = signatures(memberships(KNN, CODES, 2), tau=0.1)
    second = np.array([0, 1, 1, 1, -1])
    att = attributes(s, second, ["s1", "s2"], {"qc": np.array([1.0, 2, 3, 5, np.nan])})
    row = att.loc[[(0, 1)]].iloc[0]  # cells 0 and 3
    assert row["comp:s1"] == pytest.approx(0.5) and row["comp:s2"] == pytest.approx(0.5)
    assert row["mean:qc"] == pytest.approx(3.0)
    assert att.loc[[(1,)], "mean:qc"].item() == pytest.approx(3.0)  # cells 2, 4 (4 is NaN -> ignored)


def test_label_warnings_unlabelled_and_single_cell():
    codes, levels = label_codes(["a", "a", "b", None])
    w = " ".join(label_warnings(codes, levels))
    assert "1 unlabelled" in w and "'b' has a single cell" in w
    # a single-cell label still works end to end
    s = signatures(memberships(np.array([[1], [0], [0], [1]]), codes, 2), tau=0.1)
    assert sig_of(s, 2) == (0, 1)


def test_missing_graph_is_skipped_with_message():
    P = memberships(KNN, CODES, 2)
    cmp = compare({"A": P, "B": None}, ["a", "b"])
    assert cmp.models == ["A"]
    assert any("'B' has no kNN graph" in w for w in cmp.warnings)
    with pytest.raises(ValueError, match="no model has a kNN graph"):
        compare({"B": None}, ["a", "b"])


def test_difference_scores():
    cmp = compare({"A": memberships(KNN, CODES, 2), "B": memberships(KNN[:, ::-1], CODES, 2)}, ["a", "b"])
    fz = cmp.fuzzy.copy()
    fz["B"] = [0.0, 3.0, 1.0]
    assert difference(fz).tolist() == pytest.approx((fz.max(axis=1) - fz.min(axis=1)).tolist())
    lr = difference(fz, "log_ratio", pair=("B", "A"), pseudocount=1)
    assert lr.iloc[1] == pytest.approx(np.log2(4 / (1 + fz.A.iloc[1])))
    with pytest.raises(ValueError):
        difference(fz, "log_ratio")


# ---- synthetic two-model case ----------------------------------------------------------------------

def synthetic(k=15, seed=0):
    """300 cells: nasal / non_nasal / other (100 each). Model A: each label is its own group in the kNN graph.
    Model B: nasal and non_nasal are mixed; other stays separate."""
    rng = np.random.default_rng(seed)
    labels = np.array(["nasal"] * 100 + ["non_nasal"] * 100 + ["other"] * 100)
    codes, levels = label_codes(labels)

    def graph(groups):
        knn = np.empty((300, k), np.int64)
        for g in groups:
            for i in g:
                knn[i] = rng.choice(g[g != i], k, replace=False)
        return knn

    A = graph([np.arange(0, 100), np.arange(100, 200), np.arange(200, 300)])
    B = graph([np.arange(0, 200), np.arange(200, 300)])
    return codes, levels, {"A": memberships(A, codes, 3), "B": memberships(B, codes, 3)}


def test_synthetic_mixing_detected_and_ranked_first():
    codes, levels, Ps = synthetic()
    cmp = compare(Ps, levels, tau=0.1, max_size=3)
    key = (levels.index("nasal"), levels.index("non_nasal"))
    fz = cmp.fuzzy.loc[[key]].iloc[0]
    assert fz["A"] == 0.0            # separate groups: no cell sees >= 10% of the other label
    assert fz["B"] > 60              # 200 mixed cells, min membership ~0.5 each
    r = ranked(cmp)                  # default: by range, pure intersections hidden
    assert r.labels.iloc[0] == key and r.names.iloc[0] == "nasal & non_nasal"
    r2 = ranked(cmp, "log_ratio", pair=("B", "A"))
    assert r2.labels.iloc[0] == key and r2.score.iloc[0] > 5
    assert (levels.index("other"),) not in set(ranked(cmp).labels)  # pure hidden by default
    assert not ranked(cmp, hide_pure=False).empty


# ---- parity with the browser implementation (compute/fuzzyUpset.ts, run under Node) -------------------

@pytest.mark.parametrize("k", [15, 9])
def test_browser_parity(tmp_path, k):
    import base64
    import json
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    rng = np.random.default_rng(k)
    N, n_models = 400, 3
    cols = []
    for name, L in [("ct", 5), ("batch", 3)]:
        codes = rng.integers(0, L, N).astype(np.int32)
        codes[rng.random(N) < 0.05] = -1  # unlabelled cells
        cols.append((name, codes, L))
    counts, Ps = [], []
    for m in range(n_models):
        # blocky graphs with model-specific mixing
        blk = np.arange(N) * 4 // N
        knn = np.where(rng.random((N, k)) < 0.2 * m, rng.integers(0, N, (N, k)),
                       blk[:, None] * (N // 4) + rng.integers(0, N // 4, (N, k)))
        counts.append(np.concatenate([membership_counts(knn, c, L) for _, c, L in cols], axis=1))
        Ps.append([memberships(knn, c, L) for _, c, L in cols])
    offs = np.cumsum([0] + [L for _, _, L in cols])
    info = {"k": k, "n_levels": int(offs[-1]), "models": [f"m{m}" for m in range(n_models)],
            "columns": [{"name": n, "levels": [str(i) for i in range(L)], "offset": int(offs[i]),
                         "freq": [float(x) for x in label_frequency(c, L)]} for i, (n, c, L) in enumerate(cols)]}
    cases = [{"col": ci, "tau": tau, "max_size": ms, "normalize": nz}
             for ci in range(len(cols)) for tau in [0.1, 0.2, 0.3, 1 / (k + 1), 1.0] for ms in [1, 3] for nz in [False, True]]
    fx = tmp_path / "fixture.json"
    fx.write_text(json.dumps({"N": N, "info": info, "cases": cases,
                              "counts": base64.b64encode(np.stack(counts).tobytes()).decode()}))
    runner = Path(__file__).with_name("fuzzy_parity_runner.mjs")
    res = subprocess.run([node, "--experimental-strip-types", "--no-warnings", str(runner), str(fx)],
                         capture_output=True, text=True, check=True)
    js = json.loads(res.stdout)
    for cs, out in zip(cases, js):
        _, codes, L = cols[cs["col"]]
        freq = label_frequency(codes, L)
        per = {}
        for m in range(n_models):
            s = signatures(Ps[m][cs["col"]], cs["tau"], cs["max_size"], cs["normalize"], freq)
            per[f"m{m}"] = s
            py_sig = [",".join(map(str, s.intersections[x])) if x >= 0 else int(x) for x in s.sig]
            assert py_sig == out["models"][m]["sig"], cs
            assert np.allclose(s.strength, out["models"][m]["strength"]), cs
            assert (s.n_none, s.n_diffuse) == (out["models"][m]["n_none"], out["models"][m]["n_diffuse"])
        fz = compare({m: Ps[int(m[1:])][cs["col"]] for m in per}, info["columns"][cs["col"]]["levels"],
                     cs["tau"], cs["max_size"], cs["normalize"], freq).fuzzy
        py = {",".join(map(str, t)): fz.loc[[t]].iloc[0].tolist() for t in fz.index}
        assert py.keys() == out["fuzzy"].keys(), cs
        for key in py:
            assert np.allclose(py[key], out["fuzzy"][key]), (cs, key)
