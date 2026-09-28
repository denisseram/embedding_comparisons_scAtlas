"""Fuzzy UpSet computation on hand-built graphs with known answers, plus the synthetic two-model case."""
import numpy as np
import pytest
import scipy.sparse as sp

from pipeline.fuzzy_upset import (DIFFUSE, NONE, aggregate, attributes, cells_for, cells_in_regions, compare,
                                  difference, label_codes, label_frequency, label_warnings, membership_counts,
                                  memberships, memberships_from_adjacency, mixed_cells, mixed_regions, ranked,
                                  signatures)

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
    assert s.strength[0] == pytest.approx(1 / 3)  # diffuse cells keep the min over all their labels
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


# ---- mixed regions: a heterogeneous cluster becomes one set --------------------------------------------

def ring(start, n, k=2):
    """Neighbours i+1..i+k (mod n) inside a block of n cells starting at `start`."""
    return [[start + (i + j) % n for j in range(1, k + 1)] for i in range(n)]


def blob_graph():
    """Cells 0-19 label 0 and 20-39 label 1 (pure groups); 40-51 a 'blob' cycling labels 0,1,2,3.
    Every blob cell sees itself + its 2 ring neighbours = 3 different labels, so per-cell signatures split
    the blob into 4 different three-label intersections."""
    codes = np.array([0] * 20 + [1] * 20 + [0, 1, 2, 3] * 3, np.int32)
    knn = np.array(ring(0, 20) + ring(20, 20) + ring(40, 12))
    return knn, codes


def test_per_cell_signatures_fragment_a_heterogeneous_cluster():
    knn, codes = blob_graph()
    s = signatures(memberships(knn, codes, 4), tau=0.1, max_size=3)
    blob_sigs = {s.intersections[x] for x in s.sig[40:]}
    assert blob_sigs == {(0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3)}


def test_mixed_regions_group_the_cluster():
    knn, codes = blob_graph()
    rg = mixed_regions(knn, memberships(knn, codes, 4), tau=0.1)
    assert len(rg.n_cells) == 1 and rg.n_cells[0] == 12
    assert rg.signature == [(0, 1, 2, 3)]                      # pooled: 0.25 of each label
    assert np.allclose(rg.composition[0], 0.25)
    assert rg.fuzzy_size[0] == pytest.approx(12 / 3)           # each cell: min of three 1/3 memberships
    assert (rg.region[:40] == -1).all() and (rg.region[40:] == 0).all()
    assert cells_in_regions(rg, (3, 2, 1, 0)).tolist() == list(range(40, 52))


def test_mixed_regions_min_size_and_order():
    knn, codes = blob_graph()
    P = memberships(knn, codes, 4)
    rg = mixed_regions(knn, P, tau=0.1, min_size=13)
    assert len(rg.n_cells) == 0 and rg.n_small == 12 and (rg.region == -1).all()
    # two blobs of different size -> two regions, largest first
    codes2 = np.concatenate([codes, np.array([0, 1] * 3, np.int32)])
    knn2 = np.vstack([knn, np.array(ring(52, 6))])
    rg2 = mixed_regions(knn2, memberships(knn2, codes2, 4), tau=0.1)
    assert rg2.n_cells.tolist() == [12, 6] and rg2.signature[1] == (0, 1)
    assert (rg2.region[52:] == 1).all()


def test_mixed_regions_threshold():
    knn, codes = blob_graph()
    # at tau = 0.4 no blob cell has two labels >= 0.4 (all memberships are 1/3): no mixed cells
    rg = mixed_regions(knn, memberships(knn, codes, 4), tau=0.4)
    assert len(rg.n_cells) == 0 and rg.n_small == 0


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
    counts, Ps, knns = [], [], []
    for m in range(n_models):
        # blocky graphs with model-specific mixing
        blk = np.arange(N) * 4 // N
        knn = np.where(rng.random((N, k)) < 0.2 * m, rng.integers(0, N, (N, k)),
                       blk[:, None] * (N // 4) + rng.integers(0, N // 4, (N, k)))
        knns.append(knn)
        counts.append(np.concatenate([membership_counts(knn, c, L) for _, c, L in cols], axis=1))
        Ps.append([memberships(knn, c, L) for _, c, L in cols])
    offs = np.cumsum([0] + [L for _, _, L in cols])
    taus = [0.1, 0.2, 0.3]
    info = {"k": k, "n_levels": int(offs[-1]), "models": [f"m{m}" for m in range(n_models)],
            "columns": [{"name": n, "levels": [str(i) for i in range(L)], "offset": int(offs[i]),
                         "freq": [float(x) for x in label_frequency(c, L)]} for i, (n, c, L) in enumerate(cols)],
            "regions": {"file": "", "taus": taus, "min_size": 3}}
    cases = [{"col": ci, "tau": tau, "max_size": ms, "normalize": nz}
             for ci in range(len(cols)) for tau in [0.1, 0.2, 0.3, 1 / (k + 1), 1.0] for ms in [1, 3] for nz in [False, True]]
    reg = np.full((len(taus), len(cols), n_models, N), -1, np.int16)
    rgs = {}
    for t, tau in enumerate(taus):
        for ci in range(len(cols)):
            for m in range(n_models):
                rgs[t, ci, m] = mixed_regions(knns[m], Ps[m][ci], tau, min_size=3)
                reg[t, ci, m] = rgs[t, ci, m].region
    region_cases = [{"col": ci, "tau_idx": t} for t in range(len(taus)) for ci in range(len(cols))]
    fx = tmp_path / "fixture.json"
    fx.write_text(json.dumps({"N": N, "info": info, "cases": cases, "region_cases": region_cases,
                              "regions": base64.b64encode(reg.astype("<i2").tobytes()).decode(),
                              "counts": base64.b64encode(np.stack(counts).tobytes()).decode()}))
    runner = Path(__file__).with_name("fuzzy_parity_runner.mjs")
    res = subprocess.run([node, "--experimental-strip-types", "--no-warnings", str(runner), str(fx)],
                         capture_output=True, text=True, check=True)
    js = json.loads(res.stdout)
    # mixed regions: same per-cell region signature and strength
    for rc, out in zip(region_cases, js["regions"]):
        for m in range(n_models):
            rg = rgs[rc["tau_idx"], rc["col"], m]
            py_sig = [",".join(map(str, rg.signature[r])) if r >= 0 else NONE for r in rg.region]
            assert py_sig == out[m]["sig"], rc
            strength = np.where(rg.region >= 0, mixed_cells(Ps[m][rc["col"]], taus[rc["tau_idx"]])[1], 0)
            assert np.allclose(strength, out[m]["strength"]), rc
            assert out[m]["n_none"] == int((rg.region < 0).sum())
    for cs, out in zip(cases, js["cases"]):
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
        if any(s.n_diffuse for s in per.values()):  # the browser adds a 'diffuse' column
            py["diffuse"] = [float(s.strength[s.sig == DIFFUSE].sum()) for s in per.values()]
        assert py.keys() == out["fuzzy"].keys(), cs
        for key in py:
            assert np.allclose(py[key], out["fuzzy"][key]), (cs, key)
