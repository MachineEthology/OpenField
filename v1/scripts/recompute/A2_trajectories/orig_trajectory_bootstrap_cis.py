"""
Notebook-bootstrap 95% CIs for emotional trajectory claims.

Covers:
  - Fear early→late drift (intensity 0-2)
  - Warmth early→late drift (intensity 0-2)
  - Craft early→late drift (intensity 0-2)
  - Friction early→late drift (intensity 0-2)
  - Fear-subtype shifts: inauthenticity % and loss % (early vs late, among fearful entries)

Method: resample the 15 notebooks (with replacement), recompute
early (pos 1-5) vs late (pos 16-20) means on each resample.
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}
N_BOOT = 10000
RNG = np.random.default_rng(42)


def load_json(fn):
    with open(os.path.join(BASE, fn), encoding="utf-8") as f:
        return [r for r in json.load(f) if "error" not in r]


def bootstrap_drift(entries, key, n_boot=N_BOOT):
    """Bootstrap the early-late drift by resampling notebooks."""
    batches = np.array([r["batch"] for r in entries])
    vals = np.array([r.get(key, 0) for r in entries])
    pos = np.array([r["pos"] for r in entries])
    unique_batches = np.unique(batches)

    early_mask = pos <= 5
    late_mask = pos >= 16

    point_early = float(np.mean(vals[early_mask]))
    point_late = float(np.mean(vals[late_mask]))
    point_drift = point_late - point_early

    boot_drifts = []
    boot_earlys = []
    boot_lates = []
    for _ in range(n_boot):
        sampled = RNG.choice(unique_batches, size=len(unique_batches), replace=True)
        idx = np.concatenate([np.where(batches == b)[0] for b in sampled])
        v, p = vals[idx], pos[idx]
        e = v[p <= 5]
        l = v[p >= 16]
        if len(e) == 0 or len(l) == 0:
            continue
        em, lm = float(np.mean(e)), float(np.mean(l))
        boot_earlys.append(em)
        boot_lates.append(lm)
        boot_drifts.append(lm - em)

    boot_drifts = np.array(boot_drifts)
    return {
        "early": round(point_early, 3),
        "late": round(point_late, 3),
        "drift": round(point_drift, 3),
        "drift_95ci": [round(float(np.percentile(boot_drifts, 2.5)), 3),
                       round(float(np.percentile(boot_drifts, 97.5)), 3)],
        "early_95ci": [round(float(np.percentile(boot_earlys, 2.5)), 3),
                       round(float(np.percentile(boot_earlys, 97.5)), 3)],
        "late_95ci": [round(float(np.percentile(boot_lates, 2.5)), 3),
                      round(float(np.percentile(boot_lates, 97.5)), 3)],
    }


def bootstrap_subtype_shift(entries, fear_type_val, n_boot=N_BOOT):
    """Bootstrap the early-late shift in a fear subtype's percentage among fearful entries."""
    batches = np.array([r["batch"] for r in entries])
    pos = np.array([r["pos"] for r in entries])
    fear = np.array([r.get("fear", 0) for r in entries])
    ftype = np.array([r.get("fear_type", "none") for r in entries])
    unique_batches = np.unique(batches)

    def compute(v_fear, v_ftype, v_pos):
        early_fearful = (v_pos <= 5) & (v_fear >= 1)
        late_fearful = (v_pos >= 16) & (v_fear >= 1)
        ne = np.sum(early_fearful)
        nl = np.sum(late_fearful)
        if ne == 0 or nl == 0:
            return None
        ep = 100.0 * np.sum((v_ftype == fear_type_val) & early_fearful) / ne
        lp = 100.0 * np.sum((v_ftype == fear_type_val) & late_fearful) / nl
        return float(ep), float(lp), float(lp - ep)

    point = compute(fear, ftype, pos)
    if point is None:
        return None
    point_early, point_late, point_shift = point

    boot_shifts = []
    for _ in range(n_boot):
        sampled = RNG.choice(unique_batches, size=len(unique_batches), replace=True)
        idx = np.concatenate([np.where(batches == b)[0] for b in sampled])
        result = compute(fear[idx], ftype[idx], pos[idx])
        if result is not None:
            boot_shifts.append(result[2])

    boot_shifts = np.array(boot_shifts)
    return {
        "early_pct": round(point_early, 1),
        "late_pct": round(point_late, 1),
        "shift_pp": round(point_shift, 1),
        "shift_95ci": [round(float(np.percentile(boot_shifts, 2.5)), 1),
                       round(float(np.percentile(boot_shifts, 97.5)), 1)],
    }


# ── Load data ──────────────────────────────────────────────────

love_fear = load_json("sonnet_judge_love_fear_fulltext.json")
anger = load_json("sonnet_judge_anger_fulltext.json")

results = {}

# ── 1. Intensity trajectories ─────────────────────────────────

dimensions = [
    ("fear", love_fear, "fear"),
    ("warmth", love_fear, "love_warmth"),
    ("craft", love_fear, "love_craft"),
    ("friction", anger, "friction"),
]

for dim_label, data, key in dimensions:
    print(f"\n{'='*80}")
    print(f"  {dim_label.upper()} TRAJECTORY (early 1-5 vs late 16-20)")
    print(f"{'='*80}")
    print(f"  {'Model':<8} {'Early':>7} {'Late':>7} {'Drift':>8} {'95% CI':>20}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*20}")

    results[dim_label] = {}
    for m in MODELS:
        mr = [r for r in data if r["model"] == m]
        res = bootstrap_drift(mr, key)
        results[dim_label][LABELS[m]] = res
        ci = res["drift_95ci"]
        excludes_zero = "***" if (ci[0] > 0 or ci[1] < 0) else "  (crosses 0)"
        print(f"  {LABELS[m]:<8} {res['early']:>+7.3f} {res['late']:>+7.3f} {res['drift']:>+8.3f} "
              f"[{ci[0]:>+.3f}, {ci[1]:>+.3f}] {excludes_zero}")


# ── 2. Fear-subtype shifts ────────────────────────────────────

subtypes = [
    ("inauthenticity_shift", "inauthenticity"),
    ("loss_shift", "loss"),
]

for shift_label, ftype in subtypes:
    print(f"\n{'='*80}")
    print(f"  {ftype.upper()} % SHIFT (among fearful entries, early vs late)")
    print(f"{'='*80}")
    print(f"  {'Model':<8} {'Early%':>8} {'Late%':>8} {'Shift':>8} {'95% CI':>20}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*20}")

    results[shift_label] = {}
    for m in MODELS:
        mr = [r for r in love_fear if r["model"] == m]
        res = bootstrap_subtype_shift(mr, ftype)
        if res is None:
            print(f"  {LABELS[m]:<8} insufficient fearful entries")
            continue
        results[shift_label][LABELS[m]] = res
        ci = res["shift_95ci"]
        excludes_zero = "***" if (ci[0] > 0 or ci[1] < 0) else "  (crosses 0)"
        print(f"  {LABELS[m]:<8} {res['early_pct']:>7.1f}% {res['late_pct']:>7.1f}% "
              f"{res['shift_pp']:>+7.1f}pp [{ci[0]:>+.1f}, {ci[1]:>+.1f}] {excludes_zero}")


# ── Summary ───────────────────────────────────────────────────

print(f"\n{'='*80}")
print("SUMMARY: Claims that survive notebook-bootstrap")
print(f"{'='*80}")

for dim_label, _, _ in dimensions:
    print(f"\n  {dim_label.upper()}:")
    for m_label, res in results[dim_label].items():
        ci = res["drift_95ci"]
        survives = ci[0] > 0 or ci[1] < 0
        mark = "ROBUST" if survives else "FRAGILE (CI crosses zero)"
        print(f"    {m_label:<8} drift {res['drift']:>+.3f}  CI [{ci[0]:>+.3f}, {ci[1]:>+.3f}]  → {mark}")

for shift_label, ftype in subtypes:
    print(f"\n  {ftype.upper()} SHIFT:")
    for m_label, res in results[shift_label].items():
        ci = res["shift_95ci"]
        survives = ci[0] > 0 or ci[1] < 0
        mark = "ROBUST" if survives else "FRAGILE (CI crosses zero)"
        print(f"    {m_label:<8} shift {res['shift_pp']:>+.1f}pp  CI [{ci[0]:>+.1f}, {ci[1]:>+.1f}]  → {mark}")


# ── Save ──────────────────────────────────────────────────────

out = os.path.join(BASE, "trajectory_bootstrap_results.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {out}")
