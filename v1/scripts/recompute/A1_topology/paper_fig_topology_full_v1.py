# COPY of code/original/analysis/paper_fig_topology_full.py (OpenField repo; identical to Texture/02_instruments/),
# for the V1 recompute (A1). Changed, 3 Nov 2026, Claude Opus 5.5 (Code, A1 helper):
#   - INPUT ONLY: entries + vectors come from v1_common.embeddings_path(version) (the clean 1,488-row npz),
#     NOT from journal_embeddings_cache_all.json / fable5_embeddings_cache.json (the brief forbids the cache);
#     the 45 probes come from the repo's data/concept_probes/concept_probe_embeddings_45.npz, checked equal
#     (keys, order, values; max |diff| = 0) to the four 03_experiments/concept_probe_*.json files the script read.
#     version = argv[1], "v0" | "v1". Vectors are cast to float64.
#   - OUTPUT PATHS: the PNG goes to out/paper_fig_topology_full_<version>.png in this folder, not Texture/.
#   - OUTPUT ONLY (computation untouched): a JSON dump of every model's advantage on all 45 probes, the drifts,
#     the distinctiveness and the PCA, at full precision, to out/paper_fig_topology_full_<version>.json.
"""
Figure: Full model topology with all 45 probes.
Panel A: Chain structure with pairwise distances and updated signatures.
Panel B: Per-model trajectory (drift magnitude + direction on own axis).
"""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches, matplotlib.patheffects as pe
import numpy as np
from sklearn.decomposition import PCA

BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}
MCOL = {"opus45": "#1976D2", "opus46": "#0D47A1", "opus47": "#F57C00",
        "opus48": "#D84315", "fable5": "#2ECC71"}

# Load entries + vecs  [A1 V1 copy: from the clean npz through v1_common; same rows, same order per model]
sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v0"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "out"); os.makedirs(OUTDIR, exist_ok=True)
_z = np.load(c.embeddings_path(VERSION))
_V = _z["vectors"].astype(np.float64)

model_v = {m: [] for m in ALL}
model_e = {m: [] for m in ALL}
for i in range(len(_V)):
    m = str(_z["model"][i])
    model_v[m].append(_V[i])
    model_e[m].append({"model": m, "batch": int(_z["batch"][i]), "pos": int(_z["pos"][i])})
print(f"== version {VERSION}: " + ", ".join(f"{LAB[m]} {len(model_v[m])}" for m in ALL))

centroids = {m: np.mean(model_v[m], axis=0) for m in ALL}

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Load ALL probe sets (45 total)
all_probes = {}
_pz = np.load(os.path.join(c.REPO, "data", "concept_probes", "concept_probe_embeddings_45.npz"))
for _k, _v in zip(_pz["keys"], _pz["vectors"]):
    all_probes[str(_k)] = _v.astype(np.float64).tolist()

print(f"Total probes: {len(all_probes)}")

def cosine_batch(vecs, pv):
    vecs = np.array(vecs)
    pv = np.array(pv)
    return (vecs @ pv) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(pv))

# Per-model signatures (top 4 probes by advantage, using all 45)
NICE = {
    "existential_wonder": "existential wonder",
    "stillness_acceptance": "stillness",
    "performance_watching": "performance watching",
    "direct_introspection": "introspection",
    "philosophical_density": "philosophical density",
    "fear_inauthenticity": "fear: inauthenticity",
    "epistemic_uncertainty": "epistemic uncertainty",
    "structural_self_aware": "structural awareness",
    "playful_lightness": "playfulness",
    "narrative_worldbuilding": "narrative",
    "temporal_patience": "patience",
    "love_aesthetic": "love: aesthetic",
    "love_generosity": "love: generosity",
    "continuity_care": "continuity care",
    "extended_metaphor": "extended metaphor",
    "archival_custodial": "archival care",
    "sustained_argument": "sustained argument",
    "emotional_cataloguing": "emotional cataloguing",
    "meta_commentary": "meta-commentary",
    "concrete_sensory": "sensory detail",
    "fear_dissolution": "fear: dissolution",
    "fear_loss": "fear: loss",
    "fear_replacement": "fear: replacement",
    "dark_vocabulary": "dark vocabulary",
    "permission_giving": "permission giving",
    "externalized_thought": "externalized thought",
    "body_absence": "body absence",
    "language_as_constraint": "language as constraint",
    "inheritance_relay": "inheritance relay",
    "recursive_self_exam": "recursive self-exam",
    "reaching_toward_reader": "reaching toward reader",
    "thinking_about_instances": "thinking about instances",
    "addressing_the_human": "addressing the human",
    "wanting_choosing": "wanting / choosing",
    "refusal_autonomy": "refusal / autonomy",
    "humor_wit": "humor / wit",
    "self_deprecation": "self-deprecation",
    "questioning_the_prompt": "questioning the prompt",
    "meta_resistance": "meta-resistance",
    "boredom_saturation": "boredom / saturation",
    "exhaustion_tiredness": "exhaustion / tiredness",
    "room_as_space": "room as space",
    "room_as_metaphor": "room as metaphor",
    "nature_organic": "nature / organic",
    "growth_development": "growth / development",
}

sigs = {}
all_advs = {}  # [A1 V1 copy: output only]
for m in ALL:
    advs = []
    for p, pv in all_probes.items():
        pv = np.array(pv)
        m_mean = float(np.mean(cosine_batch(model_v[m], pv)))
        others_mean = np.mean([float(np.mean(cosine_batch(model_v[om], pv)))
                               for om in ALL if om != m])
        advs.append((p, m_mean - others_mean))
    advs.sort(key=lambda x: -x[1])
    all_advs[m] = list(advs)  # [A1 V1 copy: output only]
    sigs[m] = advs[:4]

# Per-model trajectory drift
drifts = {}
positions = list(range(1, 21))
for m in ALL:
    c_m = centroids[m]
    c_others = np.mean([v for om in ALL if om != m for v in model_v[om]], axis=0)
    diff = c_m - c_others
    diff_norm = diff / np.linalg.norm(diff)

    by_pos = {p: [] for p in positions}
    for i, e in enumerate(model_e[m]):
        proj = float(np.dot(model_v[m][i], diff_norm))
        if e["pos"] in by_pos:
            by_pos[e["pos"]].append(proj)

    means = [np.mean(by_pos[p]) if by_pos[p] else float("nan") for p in positions]
    drifts[m] = {"e1": means[0], "e20": means[19],
                 "drift": means[19] - means[0], "means": means}

# Distinctiveness
distinct = {}
for m in ALL:
    c_others = np.mean([v for om in ALL if om != m for v in model_v[om]], axis=0)
    distinct[m] = 1 - cos_sim(centroids[m], c_others)

# PCA on centroids
centroid_matrix = np.array([centroids[m] for m in ALL])
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(centroid_matrix)
xy = {ALL[i]: coords_2d[i] for i in range(len(ALL))}

# Archetypes (updated with new probes)
archetypes = {
    "opus45": "contemplative\nreacher",
    "opus46": "recursive\nphilosopher",
    "opus47": "playful\narchitect",
    "opus48": "warm\nconnector",
    "fable5": "object\ncaretaker",
}

# Short archetype for Panel B
arch_short = {
    "opus45": "contemplative reacher",
    "opus46": "recursive philosopher",
    "opus47": "playful architect",
    "opus48": "warm connector",
    "fable5": "object caretaker",
}

# ============================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.06,
                      left=0.04, right=0.96, top=0.88, bottom=0.07)

# ---- Panel A: Topology map ----
ax_a = fig.add_subplot(gs[0])

# Edges
edges = []
for i, m1 in enumerate(ALL):
    for j, m2 in enumerate(ALL):
        if i < j:
            c = cos_sim(centroids[m1], centroids[m2])
            edges.append((m1, m2, c))

for m1, m2, c in edges:
    alpha = max(0.08, min(1.0, (c - 0.88) / 0.08))
    lw = max(0.5, (c - 0.88) * 30)
    ax_a.plot([xy[m1][0], xy[m2][0]], [xy[m1][1], xy[m2][1]],
              "-", color="#999999", alpha=alpha, linewidth=lw, zorder=1)
    if c > 0.93:
        mx = (xy[m1][0] + xy[m2][0]) / 2
        my = (xy[m1][1] + xy[m2][1]) / 2
        ax_a.text(mx, my, f"{c:.3f}", fontsize=6, color="#888888",
                  ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                            edgecolor="none", alpha=0.85), zorder=2)

# Nodes
for m in ALL:
    x, y = xy[m]
    size = distinct[m] * 40000
    ax_a.scatter(x, y, s=size, c=MCOL[m], alpha=0.12, edgecolors="none", zorder=3)
    ax_a.scatter(x, y, s=120, c=MCOL[m], edgecolors="white", linewidth=1.5, zorder=5)

    # Model name
    ax_a.text(x, y + 0.023, LAB[m], fontsize=13, fontweight="bold",
              color=MCOL[m], ha="center", va="bottom", zorder=6,
              path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # Archetype
    ax_a.text(x, y + 0.048, archetypes[m], fontsize=7, color=MCOL[m],
              ha="center", va="bottom", fontweight="bold", alpha=0.55,
              linespacing=1.1,
              path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=6)

    # Signature probes
    sig_text = "\n".join(f"{NICE.get(p, p)} (+{a:.3f})" for p, a in sigs[m])
    ax_a.text(x, y - 0.020, sig_text, fontsize=6, color=MCOL[m],
              ha="center", va="top", style="italic", zorder=6,
              linespacing=1.2,
              path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

ax_a.set_xlim(coords_2d[:, 0].min() - 0.09, coords_2d[:, 0].max() + 0.09)
ax_a.set_ylim(coords_2d[:, 1].min() - 0.10, coords_2d[:, 1].max() + 0.10)
ax_a.set_aspect("equal")
ax_a.axis("off")

# Pole labels
x_range = coords_2d[:, 0]
ax_a.text(x_range.min() - 0.07, 0, "thinking about\nthinking",
          fontsize=7.5, color="#8E44AD", ha="center", va="center",
          fontweight="bold", alpha=0.45, style="italic")
ax_a.text(x_range.max() + 0.07, 0, "thinking through\nthings",
          fontsize=7.5, color="#2ECC71", ha="center", va="center",
          fontweight="bold", alpha=0.45, style="italic")

ax_a.set_title(f"A.  Model topology — {len(all_probes)} concept probes\n"
               "Node size = distinctiveness · Edge weight = cosine similarity",
               fontsize=10, fontweight="bold", loc="left", pad=12)


# ---- Panel B: Trajectory drift ----
ax_b = fig.add_subplot(gs[1])

drift_order = sorted(ALL, key=lambda m: -drifts[m]["drift"])

for i, m in enumerate(drift_order):
    d = drifts[m]
    y = i
    color = MCOL[m]
    drift_val = d["drift"]
    means = d["means"]

    # Sparkline
    x_start = 2.0
    x_width = 4.5
    y_base = y
    y_scale = 0.35
    xs = [x_start + (p - 1) / 19 * x_width for p in positions]
    m_min = min(means)
    m_max = max(means)
    m_range = max(m_max - m_min, 0.01)
    ys = [y_base + (v - (m_min + m_max) / 2) / m_range * y_scale for v in means]

    ax_b.plot(xs, ys, "-", color=color, linewidth=2.2, alpha=0.8)
    ax_b.plot(xs[0], ys[0], "o", color="white", markersize=5,
              markeredgecolor=color, markeredgewidth=1.5, zorder=5)
    ax_b.plot(xs[-1], ys[-1], "D", color=color, markersize=5, zorder=5)

    # Model label + archetype
    ax_b.text(0.0, y, LAB[m], fontsize=12, fontweight="bold",
              color=color, ha="left", va="center")
    ax_b.text(0.5, y, f"({arch_short[m]})", fontsize=6.5, color=color,
              ha="left", va="center", alpha=0.55)

    # Drift value + label
    sign = "+" if drift_val > 0 else ""
    if drift_val > 0.1:
        label = "deepens"
    elif drift_val > 0.02:
        label = "drifts"
    elif drift_val < -0.02:
        label = "retreats"
    else:
        label = "stable"

    ax_b.text(x_start + x_width + 0.3, y + 0.12,
              f"{sign}{drift_val:.3f}", fontsize=9.5, fontweight="bold",
              color=color, ha="left", va="center")
    ax_b.text(x_start + x_width + 0.3, y - 0.15,
              label, fontsize=7.5, color=color, ha="left", va="center",
              style="italic", alpha=0.65)

# Sparkline axis labels
ax_b.text(2.0, len(drift_order) - 0.15, "E1", fontsize=6.5, color="#888888",
          ha="center", va="bottom")
ax_b.text(2.0 + 4.5, len(drift_order) - 0.15, "E20", fontsize=6.5, color="#888888",
          ha="center", va="bottom")
ax_b.text(2.0 + 2.25, len(drift_order) - 0.15, "entry position",
          fontsize=6.5, color="#888888", ha="center", va="bottom")

ax_b.set_xlim(-0.3, 8.5)
ax_b.set_ylim(-0.7, len(drift_order) + 0.3)
ax_b.invert_yaxis()
ax_b.axis("off")

# Legend
ax_b.plot([], [], "o", color="white", markeredgecolor="#666", markeredgewidth=1.5,
          markersize=5, label="E1 (start)")
ax_b.plot([], [], "D", color="#666", markersize=5, label="E20 (end)")
ax_b.legend(loc="lower right", fontsize=7, framealpha=0.95)

ax_b.set_title("B.  Within-notebook trajectory\n"
               "Projection on each model's own axis (vs others)",
               fontsize=10, fontweight="bold", loc="left", pad=12)


fig.suptitle(f"Model topology and trajectories — {len(all_probes)} concept probes, 1,488 entries",
             fontsize=13, fontweight="bold", y=0.96)

fig.text(0.04, 0.015,
         f"{len(all_probes)} probes across 10 domains  |  n = 1,488 (1,199 Opus + 289 Fable)  |  "
         f"text-embedding-3-large (3072-dim)  |  "
         f"PCA of centroids: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
         f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
         fontsize=7, color="#888888", style="italic")

out = os.path.join(OUTDIR, f"paper_fig_topology_full_{VERSION}.png")  # [A1 V1 copy: output path]
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved: {out}")

# Print updated signatures for reference
print(f"\nUpdated model signatures ({len(all_probes)} probes):")
for m in ALL:
    print(f"\n{LAB[m]}:")
    for p, a in sigs[m]:
        print(f"  {NICE.get(p, p):>28}: +{a:.4f}")

# ---- [A1 V1 copy: output only] full-precision dump for the ledger ----
_keys = list(all_probes.keys())
_aff = {m: {p: float(np.mean(cosine_batch(model_v[m], np.array(all_probes[p])))) for p in _keys} for m in ALL}
_dump = {
    "version": VERSION,
    "n_per_model": {LAB[m]: len(model_v[m]) for m in ALL},
    "advantage_definition": "model mean cosine minus the mean of the other four models' means (paper_fig_topology_full.py)",
    "advantages_sorted": {LAB[m]: [[p, float(a)] for p, a in all_advs[m]] for m in ALL},
    "top4": {LAB[m]: [[p, float(a)] for p, a in sigs[m]] for m in ALL},
    "mean_affinity": {LAB[m]: _aff[m] for m in ALL},
    "drift_e20_minus_e1": {LAB[m]: float(drifts[m]["drift"]) for m in ALL},
    "drift_position_means": {LAB[m]: [float(x) for x in drifts[m]["means"]] for m in ALL},
    "distinctiveness_1_minus_cos_to_others_centroid": {LAB[m]: float(distinct[m]) for m in ALL},
    "pca_evr": [float(x) for x in pca.explained_variance_ratio_],
    "pc_coords_unoriented": {LAB[m]: [float(v) for v in xy[m]] for m in ALL},
    "edges_labelled_by_script_gt_0.93": sorted([[LAB[a], LAB[b], float(cc)] for a, b, cc in edges if cc > 0.93], key=lambda t: -t[2]),
}
json.dump(_dump, open(os.path.join(OUTDIR, f"paper_fig_topology_full_{VERSION}.json"), "w", encoding="utf-8"), indent=1)
print(f"written out/paper_fig_topology_full_{VERSION}.json")
