"""
V1 COPY (3 Nov 2026, Claude Opus 5.5, helper B). Copied from 02_instruments/valence_analysis.py (8 June 2026),
the producer of §6.4.2.1's Opus ontological means and the L1-L2 contrast +0.145 [+0.120, +0.170].
CHANGED ONLY PATHS: input valence_judge_all.json from the repository (data/judges/, byte-identical to
Texture/03_experiments/); Journal_Deep_Dive_v3.xlsx from Texture/04_analysis/ (read-only); the plot is written
to this folder. v0 only: valence_judge_all.json is the condensed-prompt Sonnet 4.6 pass (ERRATA B2) and has no
V1 counterpart; the 40 entries of 4.6's real notebooks 8 and 10 were never scored with this prompt.

Analyze the full LLM-judge valence scores (valence_judge_all.json, n=1199).

  1. Per-model means + SD across all five dimensions.
  2. Over-arc trajectories (mean per entry position 1-20) + e1->e20 deltas, plotted.
  3. Lineage test on ontological_questioning: L1 (4.5/4.6) vs L2 (4.7/4.8),
     batch-level means, bootstrap CI on the gap.
  4. Validation: correlate each judge dimension against the xlsx features that
     should track it (general<->Warmth/Loss/Beauty, self_relation<->Self Critique,
     ontological<->Own Nature Ontology, warmth<->World/Relationship/Warmth,
     ease<->Distancing). Confirms the judge measures what we think before we
     trust the per-model story.

Run after valence_judge.py --scope all completes.
"""
import os, json, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl

BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48"]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8"}
LAB2KEY = {v: k for k, v in LAB.items()}
COL = {"opus45": "#1976D2", "opus46": "#E65100", "opus47": "#2E7D32", "opus48": "#7B1FA2"}
L1, L2 = ["opus45", "opus46"], ["opus47", "opus48"]
DIMS = ["general_valence", "self_relation", "ontological_questioning",
        "ease_vs_strain", "warmth_toward_world"]
RNG = np.random.default_rng(42)

# F follow-up (3 Nov 2026, Claude Opus 5.5): a v0|v1 switch now that the 40 condensed rows exist (decision 16)
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v0"
REPO_JUDGES = (r"C:\Users\LENOVO\Desktop\the-open-field\data\judges" if VERSION == "v0"
               else r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1\inputs\judges")
HERE = os.path.dirname(os.path.abspath(__file__))                     # V1 copy: output folder
scored = json.load(open(os.path.join(REPO_JUDGES, "valence_judge_all.json"), encoding="utf-8"))
ok = [s for s in scored if s.get("status") == "ok"]
errs = [s for s in scored if s.get("status") != "ok"]
print(f"Loaded {len(scored)} ({len(ok)} ok, {len(errs)} errors).")
if errs:
    print(f"  first error: {errs[0].get('error')}")


def rows(model=None, pos=None):
    return [s for s in ok if (model is None or s["model"] == model)
            and (pos is None or s["entry_num"] == pos)]


# ---- 1. per-model means ----------------------------------------------------
print("\n=== Per-model means (all entries) ===")
print(f"  {'Model':>5}" + "".join(f"{d.split('_')[0][:11]:>13}" for d in DIMS))
print("  " + "-" * (5 + 13 * len(DIMS)))
for m in MODELS:
    r = rows(m)
    print(f"  {LAB[m]:>5}" + "".join(f"{np.mean([s[d] for s in r]):>+13.3f}" for d in DIMS))

# ---- 2. over-arc trajectories ----------------------------------------------
print("\n=== Over-arc e1->e20 delta per dimension ===")
print(f"  {'Model':>5}" + "".join(f"{d.split('_')[0][:11]:>13}" for d in DIMS))
print("  " + "-" * (5 + 13 * len(DIMS)))
for m in MODELS:
    deltas = []
    for d in DIMS:
        e1 = [s[d] for s in rows(m, 1)]; e20 = [s[d] for s in rows(m, 20)]
        deltas.append(np.mean(e20) - np.mean(e1) if e1 and e20 else np.nan)
    print(f"  {LAB[m]:>5}" + "".join(f"{x:>+13.3f}" for x in deltas))

fig, axes = plt.subplots(1, len(DIMS), figsize=(4 * len(DIMS), 4.2), sharex=True)
for ax, d in zip(axes, DIMS):
    for m in MODELS:
        ys = [np.mean([s[d] for s in rows(m, p)]) if rows(m, p) else np.nan for p in range(1, 21)]
        ax.plot(range(1, 21), ys, "-", color=COL[m], label=LAB[m], lw=1.6)
    ax.set_title(d.replace("_", " "), fontsize=10)
    ax.axhline(0, color="gray", lw=0.6); ax.set_xlabel("entry"); ax.grid(alpha=0.3)
axes[0].set_ylabel("judge score"); axes[0].legend(fontsize=8)
plt.suptitle("Valence dimensions over the 20-entry arc (Sonnet 4.6 judge, n=15 batches)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(HERE, f"valence_arc_{VERSION}.png"), dpi=150, bbox_inches="tight")  # V1 copy: output path
plt.close()
print("  Saved: valence_arc.png")

# ---- 3. lineage test on ontological_questioning ----------------------------
print("\n=== Lineage test: ontological_questioning (L1 4.5/4.6 vs L2 4.7/4.8) ===")
def batch_means(models):
    out = []
    for m in models:
        for b in range(1, 16):
            r = [s["ontological_questioning"] for s in ok if s["model"] == m and s["batch"] == b]
            if r:
                out.append(np.mean(r))
    return np.array(out)
l1, l2 = batch_means(L1), batch_means(L2)
gap = l1.mean() - l2.mean()
boot = [RNG.choice(l1, len(l1)).mean() - RNG.choice(l2, len(l2)).mean() for _ in range(10000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"  L1 mean {l1.mean():+.3f} (n={len(l1)} batches), L2 mean {l2.mean():+.3f} (n={len(l2)})")
print(f"  gap (L1-L2) = {gap:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
      f"{'CLEAR of zero' if lo > 0 or hi < 0 else 'crosses zero'}")
print("  (Positive gap = first lineage holds its own-nature question more openly.)")

# ---- 4. validation against xlsx features -----------------------------------
print("\n=== Validation: judge dimensions vs xlsx features ===")
wb = openpyxl.load_workbook(os.path.join(BASE, "04_analysis", "Journal_Deep_Dive_v3.xlsx"), read_only=True)  # V1 copy: input path
ws = wb["Entry-Level Data"]
hdr = [c.value for c in next(ws.iter_rows(max_row=1))]
feat = {}
for r in ws.iter_rows(min_row=2, values_only=True):
    row = dict(zip(hdr, r)); mk = LAB2KEY.get(str(row["Model"]).strip())
    if mk is None:
        continue
    try:
        feat[(mk, int(row["Batch"]), int(row["Entry#"]))] = row
    except (TypeError, ValueError):
        pass

def feat_vec(name, subset):
    return np.array([feat[(s["model"], s["batch"], s["entry_num"])].get(name, np.nan) for s in subset], dtype=float)

VALID = {
    "general_valence": [("Warmth Care", "+"), ("Loss Absence", "-"), ("Beauty Aesthetics", "+")],
    "self_relation": [("Self Critique Meta", "-")],
    "ontological_questioning": [("Own Nature Ontology", "+")],
    "warmth_toward_world": [("Warmth Care", "+"), ("World Objects", "+"), ("Relationship Humans", "+")],
    "ease_vs_strain": [("Distancing", "-"), ("Performance Authenti", "-")],
}
have = [s for s in ok if (s["model"], s["batch"], s["entry_num"]) in feat]
print(f"  ({len(have)} entries have xlsx features)")
for dim, checks in VALID.items():
    jv = np.array([s[dim] for s in have])
    parts = []
    for fname, expect in checks:
        fv = feat_vec(fname, have); mask = ~np.isnan(fv)
        r = np.corrcoef(jv[mask], fv[mask])[0, 1] if mask.sum() > 2 else np.nan
        ok_sign = "OK" if (expect == "+" and r > 0.1) or (expect == "-" and r < -0.1) else "weak"
        parts.append(f"{fname}={r:+.2f}({expect},{ok_sign})")
    print(f"  {dim:>24}: " + "  ".join(parts))
print("\n  Signs matching expectation = the judge tracks the framework's independent detectors.")
