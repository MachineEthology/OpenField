"""place_new_model - place a NEW model against the frozen five-model reference of
'The Open Field' §5.2.2.3 without moving the published geometry.

Inputs
  --vectors  N x 3072 .npy : text-embedding-3-large vectors of the new model's journal entries
             (raw entry text, whole, no truncation; Schrodinger's Box protocol; silences excluded)
  --meta     .json list of N dicts {"batch": int, "pos": int, "words": int (optional)} in row order
  --reference reference_paper.npz (reproduces the published numbers) or reference_clean.npz
  --name     label for the printout
Outputs (printed + <name>_placement.json next to --vectors):
  1 chain     raw cosine of the new centroid with each published centroid (paper metric), plus the
              same after centring on the frozen five-model pooled mean (sharper; anisotropy removed)
  2 axis      coordinates in the FROZEN five-centroid PCA (PC1 = the published 'axis', oriented
              4.6 -> Fable 5), the share of the new centroid's offset that lies in that plane,
              a notebook-bootstrap interval for PC1, and (for information only) how far a six-model
              refit would rotate PC1
  3 path      self-axis drift (E20 - E1) and slope*19, axis = new centroid minus the frozen pooled
              five-model mean, with a notebook-bootstrap interval
  4 probes    top-4 advantages against the frozen five-model affinity table, #probes trailing,
              and a probe pole contrast (not in the paper) for comparison with the five
"""
import argparse, json, os, sys
import numpy as np
from sklearn.decomposition import PCA
sys.stdout.reconfigure(encoding="utf-8")
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable5"}
POLE_A = ["recursive_self_exam", "philosophical_density", "fear_inauthenticity"]
POLE_B = ["extended_metaphor", "archival_custodial", "concrete_sensory"]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--reference", required=True); ap.add_argument("--name", default="new")
    ap.add_argument("--boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    R = np.load(a.reference)
    models = [str(m) for m in R["models"]]; cents = R["centroids"]; pooled = R["pooled_mean"]
    pmean, comps, coords = R["pca_mean"], R["pca_components"], R["pc_coords"]
    keys = [str(k) for k in R["probe_keys"]]; P = R["probes"]; aff5 = R["affinity"]
    X = np.load(a.vectors).astype(float); meta = json.load(open(a.meta, encoding="utf-8"))
    assert X.shape[0] == len(meta) and X.shape[1] == cents.shape[1], "vectors/meta mismatch or wrong dimensionality"
    bat = np.array([m["batch"] for m in meta]); pos = np.array([m["pos"] for m in meta])
    rng = np.random.default_rng(a.seed)
    nbs = sorted(set(bat.tolist()))
    c = X.mean(axis=0)
    out = {"name": a.name, "n_entries": int(len(X)), "n_notebooks": len(nbs), "reference": os.path.basename(a.reference)}
    print(f"== {a.name}: {len(X)} entries in {len(nbs)} notebooks; reference {os.path.basename(a.reference)}")
    if all("words" in m for m in meta):
        w = np.array([m["words"] for m in meta]); out["median_words"] = float(np.median(w))
        print(f"   median words {np.median(w):.0f}  (published five: 4.5 324, 4.7 472, 4.8 596, Fable5 646, 4.6 1,414)")

    # 1 chain
    raw = {LAB[m]: cos(c, cents[i]) for i, m in enumerate(models)}
    cen = {LAB[m]: cos(c - pooled, cents[i] - pooled) for i, m in enumerate(models)}
    out["chain_raw_cosine"] = raw; out["chain_centred_cosine"] = cen
    print("1 chain  raw cosine:     " + "  ".join(f"{k} {v:.4f}" for k, v in raw.items())
          + f"   -> nearest {max(raw, key=raw.get)}")
    print("         centred cosine: " + "  ".join(f"{k} {v:+.4f}" for k, v in cen.items())
          + f"   -> nearest {max(cen, key=cen.get)}")

    # 2 axis (frozen PCA)
    xy = (c - pmean) @ comps.T
    off = c - pmean
    inplane = float(np.linalg.norm(xy) ** 2 / np.linalg.norm(off) ** 2)
    pcs = []
    for _ in range(a.boot):
        s = rng.choice(nbs, size=len(nbs), replace=True)
        cb = np.vstack([X[bat == b] for b in s]).mean(axis=0)
        pcs.append(float((cb - pmean) @ comps[0]))
    lo, hi = np.percentile(pcs, [2.5, 97.5])
    ranks = sorted([(float(coords[i, 0]), LAB[m]) for i, m in enumerate(models)] + [(float(xy[0]), a.name)])
    out.update({"pc1": float(xy[0]), "pc2": float(xy[1]), "pc1_ci95_notebook_boot": [float(lo), float(hi)],
                "inplane_share": inplane, "pc1_order_with_new": [n for _, n in ranks]})
    print(f"2 axis   frozen PC1 {xy[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]  PC2 {xy[1]:+.4f}   in-plane share of offset {inplane:.2f}")
    print("         published PC1: " + "  ".join(f"{LAB[m]} {coords[i,0]:+.4f}" for i, m in enumerate(models)))
    print("         order on frozen PC1 (introspective -> world-facing): " + " < ".join(n for _, n in ranks))
    six = PCA(n_components=2).fit(np.vstack([cents, c]))
    ang = float(np.degrees(np.arccos(min(1.0, abs(six.components_[0] @ comps[0])))))
    out["six_model_refit"] = {"evr": [float(v) for v in six.explained_variance_ratio_], "pc1_rotation_deg": ang}
    print(f"         (info) a six-model refit would explain {100*six.explained_variance_ratio_[0]:.1f}%/"
          f"{100*six.explained_variance_ratio_[1]:.1f}% and rotate PC1 by {ang:.1f} deg - do not refit to publish")

    # 3 self-axis path
    ax = (c - pooled) / np.linalg.norm(c - pooled)
    pj = X @ ax
    means = [float(pj[pos == p].mean()) if (pos == p).any() else float("nan") for p in range(1, 21)]
    def drift(sample):
        p1 = np.concatenate([pj[(bat == b) & (pos == 1)] for b in sample])
        p20 = np.concatenate([pj[(bat == b) & (pos == 20)] for b in sample])
        return float(p20.mean() - p1.mean()) if len(p1) and len(p20) else float("nan")
    def slope(sample):
        xs = np.concatenate([pos[bat == b] for b in sample]).astype(float)
        ys = np.concatenate([pj[bat == b] for b in sample])
        return float(np.polyfit(xs, ys, 1)[0] * 19)
    d0, s0 = drift(nbs), slope(nbs)
    bd, bs = [], []
    for _ in range(a.boot):
        s = rng.choice(nbs, size=len(nbs), replace=True)
        bd.append(drift(s)); bs.append(slope(s))
    bd = np.array(bd); bd = bd[~np.isnan(bd)]
    out.update({"drift_e20_minus_e1": d0, "drift_ci95": [float(x) for x in np.percentile(bd, [2.5, 97.5])] if len(bd) else None,
                "slope_x19": s0, "slope_ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
                "n_at_pos1": int((pos == 1).sum()), "n_at_pos20": int((pos == 20).sum()), "position_means": means})
    print(f"3 path   E20-E1 {d0:+.3f} {np.round(out['drift_ci95'], 3) if out['drift_ci95'] else ''}  slope*19 {s0:+.3f} "
          f"{np.round(out['slope_ci95'], 3)}  (n at E1 = {(pos == 1).sum()}, at E20 = {(pos == 20).sum()})")
    print("         published (paper rows): Fable5 +0.318, 4.8 +0.226, 4.6 +0.155, 4.7 +0.027, 4.5 -0.061")
    complete = [b for b in nbs if set(range(1, 21)) <= set(pos[bat == b].tolist())]
    warn = []
    if len(complete) < 10:
        warn.append(f"only {len(complete)} complete 20-entry notebooks: the published drifts rest on 13-15 per model; "
                    f"treat E20-E1 and its interval as not interpretable below ~10")
    if len(nbs) < 10:
        warn.append(f"{len(nbs)} notebooks: a 3-notebook mean of a published model spans about +/-0.03 on PC1 "
                    f"(s07); 4.5 and 4.6 (gap 0.007) cannot be separated at that n")
    if inplane < 0.5:
        warn.append(f"only {100*inplane:.0f}% of the new centroid's offset lies in the frozen PC1-PC2 plane: "
                    f"its PC1 coordinate is a shadow of a mostly off-axis position; report the share with it")
    if "median_words" in out and not (300 <= out["median_words"] <= 1500):
        warn.append("median length outside the published range (324-1,414 words): length moves entries along PC1 "
                    "within every model (r = +0.19 to +0.60); add the length-residualised check")
    out["warnings"] = warn
    for wmsg in warn:
        print("   WARNING " + wmsg)

    # 4 probes
    Xu = X / np.linalg.norm(X, axis=1, keepdims=True); Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    S = Xu @ Pn.T; aff = S.mean(axis=0)
    adv = aff - aff5.mean(axis=0)
    top = [(keys[i], float(adv[i])) for i in np.argsort(-adv)[:4]]
    ia = [keys.index(k) for k in POLE_A]; ib = [keys.index(k) for k in POLE_B]
    pole = float((S[:, ib].mean(axis=1) - S[:, ia].mean(axis=1)).mean())
    ref_pole = {LAB[m]: float(aff5[i, ib].mean() - aff5[i, ia].mean()) for i, m in enumerate(models)}
    out.update({"top4_advantage_vs_frozen_five": top, "n_probes_below_five_mean": int((adv < 0).sum()),
                "mean_affinity_level": float(aff.mean()), "five_mean_affinity_level": float(aff5.mean()),
                "probe_pole_contrast": pole, "probe_pole_contrast_reference": ref_pole})
    print("4 probes top-4 vs frozen five-model mean: " + ", ".join(f"{k} {v:+.3f}" for k, v in top)
          + f"  | below the five-model mean on {(adv < 0).sum()}/45 | mean level {aff.mean():.3f} vs five {aff5.mean():.3f}")
    print(f"         probe pole contrast {pole:+.4f}  (five: " + ", ".join(f"{k} {v:+.4f}" for k, v in ref_pole.items()) + ")")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(a.vectors)), f"{a.name}_placement.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
