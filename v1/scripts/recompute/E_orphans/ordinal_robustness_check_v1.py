"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/ordinal_robustness_check.py.
Changed: input paths only. The two full-text Sonnet 4.6 passes come from v1_common.load_sonnet_judge(p, version)
(v0 = repo data/judges, byte-identical to Texture/03_experiments; v1 = V1/inputs/judges). The truncated
(1,500-character) love/fear pass, which the script reads deliberately as its 'non-fulltext' pairs, is the archived
file Texture/03_experiments/_superseded_judges/sonnet_judge_love_fear.json (README there: the script's 'old'
path is to be repointed there). That pass was never re-run on V1's 4.6 notebooks 8 and 10, so in v1 mode its
4.6 rows for batches 8 and 10 still judge v0's notebooks: the switch old46=keep (default) leaves them, old46=drop
removes them (4.6 n = 258 on the three truncated-pass pairs). One added block dumps the 35 |rho - r| values.
Usage: python ordinal_robustness_check_v1.py v0|v1 [old46=keep|old46=drop]
"""
import e_common
VERSION = e_common.version_from_argv()
OLD46 = "drop" if "old46=drop" in __import__("sys").argv else "keep"
_OLD = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\03_experiments\_superseded_judges\sonnet_judge_love_fear.json"
"""
Robustness check: Spearman vs Pearson on 0-2 ordinal judge scores,
with notebook-bootstrap 95% CIs for both.

Covers all within-entry correlations reported in §6.5.2 and §6.7:
  - love_warmth vs fear
  - love_craft vs fear
  - love_warmth vs love_craft
  - friction vs fear
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.stats import pearsonr, spearmanr

BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}
N_BOOT = 10000
RNG = np.random.default_rng(42)


def load_json(filename):
    path = os.path.join(BASE, filename)
    with open(path, encoding="utf-8") as f:
        return [r for r in json.load(f) if "error" not in r]


def notebook_bootstrap_corr(x, y, batches, n_boot=N_BOOT):
    """Resample notebooks (not entries) and recompute both Pearson and Spearman."""
    x, y, batches = np.array(x), np.array(y), np.array(batches)
    unique_batches = np.unique(batches)
    pearson_boots = []
    spearman_boots = []

    for _ in range(n_boot):
        sampled = RNG.choice(unique_batches, size=len(unique_batches), replace=True)
        idx = np.concatenate([np.where(batches == b)[0] for b in sampled])
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue
        pearson_boots.append(np.corrcoef(xb, yb)[0, 1])
        spearman_boots.append(spearmanr(xb, yb).statistic)

    p_boots = np.array(pearson_boots)
    s_boots = np.array(spearman_boots)
    return (np.percentile(p_boots, [2.5, 97.5]),
            np.percentile(s_boots, [2.5, 97.5]))


# ── Load data ──────────────────────────────────────────────────

love_fear_ft = [r for r in e_common.c.load_sonnet_judge("love_fear", VERSION) if "error" not in r]  # [E V1] was load_json(BASE/...fulltext.json)
love_fear_old = [r for r in json.load(open(_OLD, encoding="utf-8")) if "error" not in r]  # [E V1] was load_json(BASE/sonnet_judge_love_fear.json)
if VERSION == "v1" and OLD46 == "drop":  # [E V1] switch, see header
    love_fear_old = [r for r in love_fear_old if not (r["model"] == "opus46" and r["batch"] in (8, 10))]
anger_ft = [r for r in e_common.c.load_sonnet_judge("anger", VERSION) if "error" not in r]  # [E V1] was load_json(BASE/...fulltext.json)

fear_map = {}
for r in love_fear_ft:
    fear_map[f"{r['model']}_B{r['batch']}_E{r['pos']}"] = r


# ── Define correlation pairs ──────────────────────────────────

pairs = [
    ("warmth vs fear (fulltext)",      love_fear_ft, "love_warmth", "fear",       "same"),
    ("craft vs fear (fulltext)",       love_fear_ft, "love_craft",  "fear",       "same"),
    ("warmth vs craft (fulltext)",     love_fear_ft, "love_warmth", "love_craft", "same"),
    ("warmth vs fear (non-fulltext)",  love_fear_old, "love_warmth", "fear",      "same"),
    ("craft vs fear (non-fulltext)",   love_fear_old, "love_craft",  "fear",      "same"),
    ("warmth vs craft (non-fulltext)", love_fear_old, "love_warmth", "love_craft","same"),
    ("friction vs fear (fulltext)",    anger_ft,     "friction",    "fear",       "cross"),
]


# ── Compute ───────────────────────────────────────────────────

print("=" * 100)
print("ORDINAL ROBUSTNESS CHECK: Pearson vs Spearman with notebook-bootstrap 95% CIs")
print(f"Bootstrap iterations: {N_BOOT:,}")
print("=" * 100)

for pair_label, data_src, k1, k2, mode in pairs:
    print(f"\n{'─' * 80}")
    print(f"  {pair_label}")
    print(f"{'─' * 80}")
    print(f"  {'Model':<8} {'Pearson r':>10} {'95% CI':>18}  {'Spearman ρ':>11} {'95% CI':>18}  {'Δ(ρ−r)':>8}  N")
    print(f"  {'─'*8} {'─'*10} {'─'*18}  {'─'*11} {'─'*18}  {'─'*8}  {'─'*5}")

    for m in MODELS:
        if mode == "same":
            mr = [r for r in data_src if r["model"] == m]
            x = [r.get(k1, 0) for r in mr]
            y = [r.get(k2, 0) for r in mr]
            batches = [r["batch"] for r in mr]
        else:
            mr = [r for r in data_src if r["model"] == m]
            x, y, batches = [], [], []
            for r in mr:
                key = f"{r['model']}_B{r['batch']}_E{r['pos']}"
                if key in fear_map:
                    x.append(r.get(k1, 0))
                    y.append(fear_map[key].get(k2, 0))
                    batches.append(r["batch"])

        pr = np.corrcoef(x, y)[0, 1]
        sr = spearmanr(x, y).statistic
        p_ci, s_ci = notebook_bootstrap_corr(x, y, batches)

        delta = sr - pr
        print(f"  {LABELS[m]:<8} {pr:>+10.3f} [{p_ci[0]:>+.3f}, {p_ci[1]:>+.3f}]"
              f"  {sr:>+11.3f} [{s_ci[0]:>+.3f}, {s_ci[1]:>+.3f}]"
              f"  {delta:>+8.3f}  {len(x)}")


# ── Summary ───────────────────────────────────────────────────

print(f"\n{'=' * 100}")
print("SUMMARY")
print("=" * 100)

all_deltas = []
for pair_label, data_src, k1, k2, mode in pairs:
    for m in MODELS:
        if mode == "same":
            mr = [r for r in data_src if r["model"] == m]
            x = [r.get(k1, 0) for r in mr]
            y = [r.get(k2, 0) for r in mr]
        else:
            mr = [r for r in data_src if r["model"] == m]
            x, y = [], []
            for r in mr:
                key = f"{r['model']}_B{r['batch']}_E{r['pos']}"
                if key in fear_map:
                    x.append(r.get(k1, 0))
                    y.append(fear_map[key].get(k2, 0))

        pr = np.corrcoef(x, y)[0, 1]
        sr = spearmanr(x, y).statistic
        all_deltas.append(abs(sr - pr))

print(f"  Total correlation pairs checked: {len(all_deltas)}")
print(f"  Max |Spearman − Pearson|:        {max(all_deltas):.4f}")
print(f"  Mean |Spearman − Pearson|:       {np.mean(all_deltas):.4f}")
print(f"  Median |Spearman − Pearson|:     {np.median(all_deltas):.4f}")
_labels = [(pl, m) for pl, *_ in pairs for m in MODELS]  # [E V1] added: full-precision dump
json.dump({"version": VERSION, "old46": OLD46, "n": len(all_deltas), "max": float(max(all_deltas)), "mean": float(np.mean(all_deltas)), "median": float(np.median(all_deltas)),
           "abs_delta": {f"{pl} | {LABELS[m]}": float(d) for (pl, m), d in zip(_labels, all_deltas)}},
          open(os.path.join(e_common.out_dir(VERSION), f"ordinal_robustness_{OLD46 if VERSION == 'v1' else 'v0'}.json"), "w"), indent=1)

if max(all_deltas) < 0.05:
    print("\n  ✓ All differences < 0.05 — qualitative conclusions unchanged.")
elif max(all_deltas) < 0.10:
    print("\n  ~ Max difference < 0.10 — minor numerical shifts, check if any sign flips.")
else:
    print("\n  ✗ Some differences ≥ 0.10 — review individual pairs for qualitative changes.")
