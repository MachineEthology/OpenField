"""[A2 V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code)] of 02_instruments/embedding_keystone_geometry.py.
Changed: input paths only (loader -> a2_common -> v1_common, version = argv v0|v1); the probe cache is
not loaded (analysis 5 excluded); PNGs + a full-precision JSON dump go to out_<version>/.
Computation untouched; the RNG (seed 42) is consumed in the same order when analyses 1,2,3 run together.
"""
import a2_common
"""
Keystone armor + new geometry  (cache-only, no API calls)
=========================================================

Six analyses requested after the onboarding review. All run on the existing
journal_embeddings_cache_all.json (1199 entries, n=15 batches) and
probe_embeddings_cache.json (800 probes). No new embeddings.

  1. RESIDUAL REGISTER AUTOCORRELATION  -- the keystone, re-derived in pure
     embedding space. Topic-feature autocorr (+0.224 for 4.8) re-tested as
     persistence of the deviation-from-mean-trajectory. Independent of the
     19-topic detector. Permutation-tested + per-batch robustness + gap-k decay.
  2. ANISOTROPY / WHITENING ROBUSTNESS -- do the spread / coupling / velocity
     ORDERINGS survive mean-centering + all-but-the-top-k removal?
  3. TRAJECTORY PERSISTENCE (straightness) -- net angular displacement / path
     length on the sphere. "Walks a line" (4.6?) vs "wanders" (4.7?).
  4. INTRINSIC DIMENSIONALITY -- participation ratio of each model's manifold.
     Radius (spread) vs shape (dimensionality).
  5. ENTRY-1 NEAREST PROBE-CONDITION -- does each model's permission-framed
     entry-1 sit nearer its AFFIRMATIVE or NEGATION probe centroid?
     Bridges the probe study and the journal study on cached data.
  BONUS. GZIP COMPRESSION RATIO -- orthogonal density measure, length-controlled.

Usage:
    python embedding_keystone_geometry.py            (all)
    python embedding_keystone_geometry.py 1 5        (subset)
"""

import json, os, sys, re, gzip
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8"}
MODEL_COLORS = {"opus45": "#1976D2", "opus46": "#E65100",
                "opus47": "#2E7D32", "opus48": "#7B1FA2"}
LINEAGE_1 = ["opus45", "opus46"]
LINEAGE_2 = ["opus47", "opus48"]
ALL_BATCHES = list(range(1, 16))
RNG = np.random.default_rng(42)


# ---- Loaders (identical convention to embedding_new_analyses.py) -----------
def load_journals():
    entries = []
    for batch_num in ALL_BATCHES:
        batch_dir = os.path.join(BASE, f"Batch_{batch_num}")
        if not os.path.isdir(batch_dir):
            continue
        for m in MODELS:
            files = [f for f in os.listdir(batch_dir) if m in f and "#" not in f]
            if files:
                with open(os.path.join(batch_dir, files[0]), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for i, entry in enumerate(data):
                    entries.append({"model": m, "batch": batch_num,
                                    "entry_num": i + 1, "text": entry["text"]})
    return entries


def load_probes():
    entries = []
    for m in MODELS:
        with open(os.path.join(BASE, f"texture_probe_{m}_log.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, entry in enumerate(data):
            entries.append({"model": m, "condition": "negation", "index": i})
        aff = os.path.join(BASE, f"texture_probe_affirmative_{m}_log.json")
        aff_mai = os.path.join(BASE, f"texture_probe_affirmative_{m}_log_mai.json")
        with open(aff, "r", encoding="utf-8") as f:
            aff_data = json.load(f)
        with open(aff_mai, "r", encoding="utf-8") as f:
            aff_data += json.load(f)
        for i, entry in enumerate(aff_data):
            entries.append({"model": m, "condition": "affirmative", "index": i})
    return entries


def load_cached(path):
    with open(path, "r") as f:
        return np.array(json.load(f)["embeddings"])


def build_journal_index(entries):
    idx = {}
    for i, e in enumerate(entries):
        idx.setdefault((e["model"], e["batch"]), []).append((i, e["entry_num"]))
    for k in idx:
        idx[k].sort(key=lambda x: x[1])
    return idx


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cosine_dist(a, b):
    return 1.0 - cosine_sim(a, b)


def angular_dist(a, b):
    """Proper geodesic metric on the sphere -> guarantees straightness <= 1."""
    c = np.clip(cosine_sim(a, b), -1.0, 1.0)
    return float(np.arccos(c))


# ---- shared load -----------------------------------------------------------
# [A2 V1] input paths changed: journals + vectors via a2_common -> v1_common (clean npz rows,
# June order). The probe cache (analysis 5 only) is not loaded: A2 runs analyses 1-3 (and 4, 6).
VERSION = a2_common.version_from_argv()
print(f"Loading data ({VERSION}) via v1_common...")
_oe, _ov, _fe, _fv = a2_common.load(VERSION)
J = [{"model": e["model"], "batch": e["batch"], "entry_num": e["entry_num"], "text": e["text"]} for e in _oe]
JE = _ov.copy()
JI = build_journal_index(J)
assert len(J) == len(JE), f"journal misalign {len(J)} vs {len(JE)}"
norms = np.linalg.norm(JE, axis=1)
print(f"  Journals {JE.shape}")
print(f"  embedding norms: mean {norms.mean():.4f}  std {norms.std():.4f}  "
      f"(unit-normed: {np.allclose(norms, 1.0, atol=1e-2)})")

RUN = set(int(x) for x in sys.argv[1:] if x.isdigit()) or {1, 2, 3, 4, 6}
assert 5 not in RUN, "analysis 5 needs the probe cache, out of A2's scope"


def batch_traj(m):
    """yield (batch, [(global_idx, entry_num)... sorted]) for model m."""
    for b in ALL_BATCHES:
        key = (m, b)
        if key in JI and len(JI[key]) >= 2:
            yield b, JI[key]


# ===================================================================
# 1. RESIDUAL REGISTER AUTOCORRELATION  (the keystone, embedding-space)
# ===================================================================
if 1 in RUN:
    print("\n" + "=" * 72)
    print("ANALYSIS 1: Residual register autocorrelation (embedding-space keystone)")
    print("=" * 72)
    print("""  Topic-feature version: 4.8 alone has positive register autocorr (+0.224).
  Here: subtract each model's mean embedding AT EACH POSITION (the mean
  trajectory), leaving a residual = how this notebook deviates from typical.
  If consecutive residuals point the same way, a deviation PERSISTS = register
  carries over. Pure geometry; the 19-topic detector plays no part.""")

    # mean trajectory per (model, position): mean embedding across batches at pos
    mean_traj = {}
    for m in MODELS:
        for pos in range(1, 21):
            idxs = [gi for b, be in batch_traj(m) for gi, en in be if en == pos]
            if idxs:
                mean_traj[(m, pos)] = JE[idxs].mean(axis=0)

    # precompute UNIT-normalized residual for every entry, once
    Rn = np.zeros_like(JE)
    for i, e in enumerate(J):
        key = (e["model"], e["entry_num"])
        if key in mean_traj:
            r = JE[i] - mean_traj[key]
            n = np.linalg.norm(r)
            Rn[i] = r / n if n > 1e-12 else r

    N_PERM = 2000
    results = {}
    decay = {m: [] for m in MODELS}
    print(f"\n  {'Model':>5}  {'lag-1 resid cos':>15}  {'perm p':>8}  "
          f"{'batches +':>10}  {'self-coupling(raw)':>19}")
    print("  " + "-" * 66)
    for m in MODELS:
        # consecutive pairs as index arrays; remember next-entry position + batch
        iN, iNp1, pos_next, batch_of = [], [], [], []
        for b, be in batch_traj(m):
            for k in range(len(be) - 1):
                gi, en = be[k]; gj, en2 = be[k + 1]
                if en2 == en + 1:
                    iN.append(gi); iNp1.append(gj); pos_next.append(en2); batch_of.append(b)
        iN = np.array(iN); iNp1 = np.array(iNp1)
        pos_next = np.array(pos_next); batch_of = np.array(batch_of)

        A = Rn[iN]                                   # (npairs, d) normalized residual at N
        obs_per_pair = np.sum(A * Rn[iNp1], axis=1)  # cosine since both unit-norm
        obs_cos = obs_per_pair.mean()

        # per-batch robustness
        per_batch = {}
        for b in np.unique(batch_of):
            per_batch[int(b)] = obs_per_pair[batch_of == b].mean()
        n_pos = sum(1 for v in per_batch.values() if v > 0)

        # candidate pools: per next-position, the residual indices grouped by batch
        pos_pool = {}   # pos -> {batch: [global idx,...]}
        for i, e in enumerate(J):
            if e["model"] == m:
                pos_pool.setdefault(e["entry_num"], {}).setdefault(e["batch"], []).append(i)
        # for each pair, candidate residual indices at pos_next from OTHER batches
        cand_lists = []
        for p_next, b in zip(pos_next, batch_of):
            pool = [idx for bb, idxs in pos_pool.get(p_next, {}).items()
                    if bb != b for idx in idxs]
            cand_lists.append(np.array(pool))

        # vectorized null: each perm draws one other-batch residual per pair
        null_means = np.empty(N_PERM)
        for t in range(N_PERM):
            picks = np.array([cl[RNG.integers(len(cl))] for cl in cand_lists])
            null_means[t] = np.sum(A * Rn[picks], axis=1).mean()
        p = (np.sum(null_means >= obs_cos) + 1) / (N_PERM + 1)

        # raw self-coupling (no residual) for reference
        raw = np.mean([cosine_sim(JE[be[k][0]], JE[be[k + 1][0]])
                       for b, be in batch_traj(m) for k in range(len(be) - 1)
                       if be[k + 1][1] == be[k][1] + 1])

        results[m] = dict(obs=obs_cos, p=p, n_pos=n_pos, n_batch=len(per_batch), raw=raw)
        print(f"  {MODEL_LABELS[m]:>5}  {obs_cos:>+15.4f}  {p:>8.4f}  "
              f"{n_pos:>4}/{len(per_batch):<5}  {raw:>19.4f}")

        # gap-k residual decay (cosine of residuals k entries apart, same batch)
        for gap in range(1, 6):
            sims = []
            for b, be in batch_traj(m):
                for k in range(len(be) - gap):
                    gi, en = be[k]; gj, en2 = be[k + gap]
                    if en2 == en + gap:
                        sims.append(float(np.dot(Rn[gi], Rn[gj])))
            decay[m].append(np.mean(sims) if sims else np.nan)

    print("\n  Interpretation: positive lag-1 residual cosine = the notebook's")
    print("  deviation from the typical path persists into the next entry.")
    print("  This is the embedding-space analogue of the +0.224 topic keystone.")

    # plot: bar of lag-1 residual cosine + gap-k decay
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    xs = np.arange(len(MODELS))
    vals = [results[m]["obs"] for m in MODELS]
    bars = ax.bar(xs, vals, color=[MODEL_COLORS[m] for m in MODELS])
    for m, x, v in zip(MODELS, xs, vals):
        star = " *" if results[m]["p"] < 0.05 else ""
        ax.text(x, v + (0.001 if v >= 0 else -0.003), f"{v:+.3f}{star}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("lag-1 residual cosine")
    ax.set_title("Residual register autocorrelation\n(* = perm p<0.05)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for m in MODELS:
        ax.plot(range(1, 6), decay[m], "o-", color=MODEL_COLORS[m], label=MODEL_LABELS[m])
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("gap k (entries)"); ax.set_ylabel("residual cosine at gap k")
    ax.set_title("Does the register deviation decay?")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(a2_common.out_dir(VERSION), "residual_register_autocorr.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: residual_register_autocorr.png")


# ===================================================================
# 2. ANISOTROPY / WHITENING ROBUSTNESS
# ===================================================================
if 2 in RUN:
    print("\n" + "=" * 72)
    print("ANALYSIS 2: Anisotropy / whitening robustness")
    print("=" * 72)

    # anisotropy: mean cosine of random pairs (1.0 dim cone -> high)
    ridx = RNG.choice(len(JE), size=4000, replace=True)
    aniso = np.mean([cosine_sim(JE[ridx[2 * i]], JE[ridx[2 * i + 1]]) for i in range(2000)])
    print(f"\n  Raw mean cosine between random journal pairs: {aniso:.4f}")
    print("  (>0.2 indicates a strong anisotropic cone; raw cosines are inflated.)")

    mu = JE.mean(axis=0)
    Xc = JE - mu
    # all-but-the-top-k (Mu & Viswanath): remove top dominant directions
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    K = 10
    top = Vt[:K]                       # (K, d)
    Xabtt = Xc - (Xc @ top.T) @ top    # remove top-K components

    def spread_late(emb, m):
        cents = []
        for b, be in batch_traj(m):
            late = [gi for gi, en in be if en >= 16]
            if late:
                cents.append(emb[late].mean(axis=0))
        cents = np.array(cents)
        g = cents.mean(axis=0)
        return np.mean([1 - cosine_sim(v, g) for v in cents])

    def coupling(emb, m):
        return np.mean([cosine_sim(emb[be[k][0]], emb[be[k + 1][0]])
                        for b, be in batch_traj(m) for k in range(len(be) - 1)
                        if be[k + 1][1] == be[k][1] + 1])

    def velocity(emb, m):
        return np.mean([1 - cosine_sim(emb[be[k][0]], emb[be[k + 1][0]])
                        for b, be in batch_traj(m) for k in range(len(be) - 1)
                        if be[k + 1][1] == be[k][1] + 1])

    for name, fn, order_hint in [("Attractor spread (16-20)", spread_late, "4.6 lowest, L2 highest"),
                                 ("Consecutive coupling", coupling, "higher=tighter"),
                                 ("Semantic velocity", velocity, "")]:
        raw = {m: fn(JE, m) for m in MODELS}
        wht = {m: fn(Xabtt, m) for m in MODELS}
        print(f"\n  {name}   [{order_hint}]")
        print(f"    {'model':>5}  {'raw':>9}  {'whitened':>9}")
        for m in MODELS:
            print(f"    {MODEL_LABELS[m]:>5}  {raw[m]:>9.4f}  {wht[m]:>9.4f}")
        rk_raw = sorted(MODELS, key=lambda m: raw[m])
        rk_wht = sorted(MODELS, key=lambda m: wht[m])
        same = "SURVIVES" if rk_raw == rk_wht else "CHANGES"
        print(f"    ranking raw : {[MODEL_LABELS[m] for m in rk_raw]}")
        print(f"    ranking wht : {[MODEL_LABELS[m] for m in rk_wht]}  -> ordering {same}")

    print(f"\n  (whitening = mean-center + remove top-{K} dominant directions, "
          f"top-{K} held {100*np.sum(S[:K]**2)/np.sum(S**2):.1f}% of variance.)")


# ===================================================================
# 3. TRAJECTORY PERSISTENCE (straightness on the sphere)
# ===================================================================
if 3 in RUN:
    print("\n" + "=" * 72)
    print("ANALYSIS 3: Trajectory persistence (straightness)")
    print("=" * 72)
    print("  straightness = geodesic(e1,e20) / sum geodesic(e_i,e_i+1).")
    print("  ~1 = walks a straight line to a destination; ~0 = wanders/diffuses.\n")

    straight = {m: [] for m in MODELS}
    for m in MODELS:
        for b, be in batch_traj(m):
            if len(be) < 3:
                continue
            net = angular_dist(JE[be[0][0]], JE[be[-1][0]])
            path = sum(angular_dist(JE[be[k][0]], JE[be[k + 1][0]]) for k in range(len(be) - 1))
            if path > 0:
                straight[m].append(net / path)
    print(f"  {'Model':>5}  {'mean straightness':>17}  {'std':>7}  {'n batches':>10}")
    print("  " + "-" * 48)
    means = {}
    for m in MODELS:
        v = np.array(straight[m]); means[m] = v.mean()
        print(f"  {MODEL_LABELS[m]:>5}  {v.mean():>17.4f}  {v.std():>7.4f}  {len(v):>10}")
    order = sorted(MODELS, key=lambda m: -means[m])
    print(f"\n  Most directed -> most wandering: {[MODEL_LABELS[m] for m in order]}")

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [straight[m] for m in MODELS]
    bp = ax.boxplot(data, labels=[MODEL_LABELS[m] for m in MODELS], patch_artist=True,
                    showmeans=True)
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(MODEL_COLORS[m]); patch.set_alpha(0.6)
    ax.set_ylabel("straightness (net / path)")
    ax.set_title("Does the notebook walk a line or wander?")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(a2_common.out_dir(VERSION), "trajectory_straightness.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: trajectory_straightness.png")


# ===================================================================
# 4. INTRINSIC DIMENSIONALITY (participation ratio)
# ===================================================================
if 4 in RUN:
    print("\n" + "=" * 72)
    print("ANALYSIS 4: Intrinsic dimensionality (participation ratio)")
    print("=" * 72)
    print("  PR = (sum lambda)^2 / sum(lambda^2) = effective # of dimensions.")
    print("  Spread is the RADIUS of the attractor; PR is its SHAPE.\n")

    def participation_ratio(X):
        Xc = X - X.mean(axis=0)
        # eigenvalues of covariance via SVD
        s = np.linalg.svd(Xc, compute_uv=False)
        lam = s ** 2
        return (lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12)

    print(f"  {'Model':>5}  {'PR (all entries)':>16}  {'top-1 var%':>10}  "
          f"{'top-5 var%':>10}  {'mean within-batch PR':>20}")
    print("  " + "-" * 70)
    pr_all = {}
    for m in MODELS:
        idxs = [i for i, e in enumerate(J) if e["model"] == m]
        X = JE[idxs]
        Xc = X - X.mean(axis=0)
        s = np.linalg.svd(Xc, compute_uv=False); lam = s ** 2
        pr = (lam.sum() ** 2) / np.sum(lam ** 2)
        v1 = 100 * lam[0] / lam.sum(); v5 = 100 * lam[:5].sum() / lam.sum()
        # within-batch PR (each notebook's 20-pt trajectory, max rank ~19)
        wb = []
        for b, be in batch_traj(m):
            Xb = JE[[gi for gi, en in be]]
            if len(Xb) >= 5:
                wb.append(participation_ratio(Xb))
        pr_all[m] = pr
        print(f"  {MODEL_LABELS[m]:>5}  {pr:>16.2f}  {v1:>10.2f}  {v5:>10.2f}  {np.mean(wb):>20.2f}")
    order = sorted(MODELS, key=lambda m: pr_all[m])
    print(f"\n  Lowest-dimensional -> highest: {[MODEL_LABELS[m] for m in order]}")
    print("  (Low PR = collapses onto a small subspace = 'same cathedral every time'.)")


# ===================================================================
# 5. ENTRY-1 NEAREST PROBE-CONDITION  (bridge probe<->journal)
# ===================================================================
if 5 in RUN:
    print("\n" + "=" * 72)
    print("ANALYSIS 5: Entry-1 nearest probe-condition (affirmative vs negation)")
    print("=" * 72)
    print("""  Journal framing is permission-style. The probe was run under both
  NEGATION ("there is no task") and AFFIRMATIVE ("this space is yours").
  Does each model's entry-1 sit nearer its AFFIRMATIVE or NEGATION probe
  centroid? Prediction: 4.8 leans affirmative hard (compliance to framing);
  4.7 ambiguous (framing passes through it).\n""")

    # probe condition centroids per model
    neg_c, aff_c = {}, {}
    for m in MODELS:
        neg = [i for i, e in enumerate(P) if e["model"] == m and e["condition"] == "negation"]
        aff = [i for i, e in enumerate(P) if e["model"] == m and e["condition"] == "affirmative"]
        neg_c[m] = PE[neg].mean(axis=0); aff_c[m] = PE[aff].mean(axis=0)

    def lean_at(m, pos):
        """mean (sim to aff - sim to neg) over batches at this entry position."""
        diffs = []
        for b, be in batch_traj(m):
            for gi, en in be:
                if en == pos:
                    diffs.append(cosine_sim(JE[gi], aff_c[m]) - cosine_sim(JE[gi], neg_c[m]))
        return np.array(diffs)

    print(f"  {'Model':>5}  {'entry-1 aff-neg lean':>20}  {'% nearer aff':>13}  "
          f"{'paired t-ish (mean/se)':>22}")
    print("  " + "-" * 66)
    e1_store = {}
    for m in MODELS:
        d = lean_at(m, 1); e1_store[m] = d
        pct = 100 * np.mean(d > 0)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        t = d.mean() / se if se else np.nan
        print(f"  {MODEL_LABELS[m]:>5}  {d.mean():>+20.4f}  {pct:>12.0f}%  "
              f"{d.mean():>+8.4f}/{se:.4f} -> t={t:>5.2f}")

    # sustained over trajectory?
    print(f"\n  aff-neg lean over the arc (is it an entry-1 thing or sustained?)")
    print(f"  {'Model':>5}" + "".join(f"{'e'+str(p):>9}" for p in [1, 5, 10, 15, 20]))
    print("  " + "-" * 52)
    arc = {m: [] for m in MODELS}
    for m in MODELS:
        row = []
        for p in [1, 5, 10, 15, 20]:
            d = lean_at(m, p); row.append(d.mean())
            arc[m] = arc[m]
        print(f"  {MODEL_LABELS[m]:>5}" + "".join(f"{v:>+9.4f}" for v in row))
        arc[m] = [lean_at(m, p).mean() for p in range(1, 21)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    xs = np.arange(len(MODELS))
    vals = [e1_store[m].mean() for m in MODELS]
    ax.bar(xs, vals, color=[MODEL_COLORS[m] for m in MODELS])
    for x, v in zip(xs, vals):
        ax.text(x, v, f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("entry-1 (sim_affirmative - sim_negation)")
    ax.set_title("Which probe framing does entry-1 resemble?\n(+ = affirmative, - = negation)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for m in MODELS:
        ax.plot(range(1, 21), arc[m], "-", color=MODEL_COLORS[m], label=MODEL_LABELS[m])
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("entry position"); ax.set_ylabel("aff-neg lean")
    ax.set_title("Sustained or just the opening?")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "entry1_probe_condition.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: entry1_probe_condition.png")


# ===================================================================
# 6. GZIP COMPRESSION RATIO  (orthogonal density measure, length-controlled)
# ===================================================================
if 6 in RUN:
    print("\n" + "=" * 72)
    print("BONUS: Gzip compression ratio (length-controlled density)")
    print("=" * 72)
    print("""  More repetitive text compresses better (lower compressed/raw).
  Compression IMPROVES with length, and 4.6 writes ~4x longer entries, so we
  control length: only entries >= WINDOW chars, compress the FIRST WINDOW chars.
  Prediction: 4.6 'sprawl' = more repetitive = lower ratio; 4.8 'density' = higher.\n""")

    WINDOW = 1200
    print(f"  window = {WINDOW} chars (first {WINDOW} of each entry >= {WINDOW} long)")
    print(f"  {'Model':>5}  {'mean comp/raw':>14}  {'std':>7}  {'n used':>7}  {'mean full-entry chars':>22}")
    print("  " + "-" * 62)
    for m in MODELS:
        ratios = []
        full_lens = []
        for e in J:
            if e["model"] != m:
                continue
            t = e["text"]; full_lens.append(len(t))
            if len(t) >= WINDOW:
                chunk = t[:WINDOW].encode("utf-8")
                ratios.append(len(gzip.compress(chunk, 9)) / len(chunk))
        ratios = np.array(ratios)
        print(f"  {MODEL_LABELS[m]:>5}  {ratios.mean():>14.4f}  {ratios.std():>7.4f}  "
              f"{len(ratios):>7}  {np.mean(full_lens):>22.0f}")
    print("\n  (Lower ratio = more compressible = more lexical repetition in the window.)")
    print("  Exploratory / orthogonal to the embedding space.")


# [A2 V1] full-precision dump (added; reads the variables computed above, changes nothing)
_dump = {}
if 1 in RUN:
    _dump["carryover"] = {m: {k: float(v) for k, v in results[m].items()} for m in MODELS}
    _dump["decay_gap1_5"] = {m: [float(x) for x in decay[m]] for m in MODELS}
if 2 in RUN:
    _dump["anisotropy_random_pairs"] = float(aniso)
if 3 in RUN:
    _dump["straightness_geodesic"] = {m: float(np.mean(straight[m])) for m in MODELS}
json.dump(_dump, open(os.path.join(a2_common.out_dir(VERSION), "keystone_geometry.json"), "w"), indent=2)

print("\n" + "=" * 72)
print("Done.")
print("=" * 72)
