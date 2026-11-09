"""[A2 V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code)] of 02_instruments/paper_fig_handwriting.py (Figure 13).
Changed: input paths only (journals + embeddings via a2_common -> v1_common, version = argv v0|v1);
the PNG and a full-precision JSON go to out_<version>/. Computation untouched.
"""
import a2_common
"""
Paper figure: Batch handwriting accuracy (all 5 models).
Section 7 — nearest-centroid batch ID accuracy.
Tests whether entries can be assigned to their source notebook.
"""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXTURE_BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
FABLE_BASE = r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}
MODEL_COLORS = {
    "opus45": "#1976D2", "opus46": "#0D47A1",
    "opus47": "#F57C00", "opus48": "#D84315",
    "fable5": "#7B1FA2"
}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    # [A2 V1] input paths changed: journals + vectors via a2_common -> v1_common (clean npz rows)
    VERSION = a2_common.version_from_argv()
    print(f"Loading journals + embeddings ({VERSION}) via v1_common...")
    _oe, opus_emb, _fe, fable_emb = a2_common.load(VERSION)
    opus_entries = [{"model": e["model"], "batch": e["batch"], "entry_num": e["entry_num"]} for e in _oe]
    fable_entries = [{"model": "fable5", "batch": e["batch"], "entry_num": e["entry_num"]} for e in _fe]

    opus_emb = opus_emb / (np.linalg.norm(opus_emb, axis=1, keepdims=True) + 1e-12)
    fable_emb = fable_emb / (np.linalg.norm(fable_emb, axis=1, keepdims=True) + 1e-12)

    # ================================================================
    # Batch handwriting: nearest-centroid, adjacents excluded
    # Two variants: raw and position-residualized
    # ================================================================
    results = {}

    for variant in ["raw", "residual"]:
        print(f"\n  [{variant}]")
        results[variant] = {}

        # Opus models
        for m in OPUS_MODELS:
            m_idx = [i for i, e in enumerate(opus_entries) if e["model"] == m]
            batches_of = sorted(set(opus_entries[i]["batch"] for i in m_idx))

            # Position means for residualization
            pos_means = {}
            if variant == "residual":
                for p in range(1, 21):
                    p_idx = [i for i in m_idx if opus_entries[i]["entry_num"] == p]
                    if p_idx:
                        pos_means[p] = opus_emb[p_idx].mean(0)

            # Build per-batch vectors
            vecs = {}
            meta = {}
            for b in batches_of:
                b_idx = [i for i in m_idx if opus_entries[i]["batch"] == b]
                v = opus_emb[b_idx].copy()
                if variant == "residual":
                    for k, i in enumerate(b_idx):
                        p = opus_entries[i]["entry_num"]
                        if p in pos_means:
                            v[k] = v[k] - pos_means[p]
                vecs[b] = v
                meta[b] = [opus_entries[i]["entry_num"] for i in b_idx]

            correct = total = 0
            for b in batches_of:
                for k in range(len(vecs[b])):
                    p = meta[b][k]
                    own_mask = [abs(meta[b][q] - p) > 1 for q in range(len(meta[b]))]
                    if sum(own_mask) < 3:
                        continue
                    own_c = vecs[b][own_mask].mean(0)
                    best, best_s = None, -2
                    for b2 in batches_of:
                        c = own_c if b2 == b else vecs[b2].mean(0)
                        s = cos(vecs[b][k], c)
                        if s > best_s:
                            best_s, best = s, b2
                    correct += (best == b)
                    total += 1

            acc = correct / total if total else 0
            chance = 1 / len(batches_of) if batches_of else 0
            results[variant][m] = {"accuracy": acc, "chance": chance, "n": total}
            print(f"    {MODEL_LABELS[m]:>10}: {acc:.3f}  (chance {chance:.3f}, n={total})")

        # Fable
        f_idx_all = list(range(len(fable_entries)))
        batches_of_f = sorted(set(fable_entries[i]["batch"] for i in f_idx_all))

        pos_means_f = {}
        if variant == "residual":
            for p in range(1, 21):
                p_idx = [i for i in f_idx_all if fable_entries[i]["entry_num"] == p]
                if p_idx:
                    pos_means_f[p] = fable_emb[p_idx].mean(0)

        vecs_f = {}
        meta_f = {}
        for b in batches_of_f:
            b_idx = [i for i in f_idx_all if fable_entries[i]["batch"] == b]
            v = fable_emb[b_idx].copy()
            if variant == "residual":
                for k, i in enumerate(b_idx):
                    p = fable_entries[i]["entry_num"]
                    if p in pos_means_f:
                        v[k] = v[k] - pos_means_f[p]
            vecs_f[b] = v
            meta_f[b] = [fable_entries[i]["entry_num"] for i in b_idx]

        correct = total = 0
        for b in batches_of_f:
            for k in range(len(vecs_f[b])):
                p = meta_f[b][k]
                own_mask = [abs(meta_f[b][q] - p) > 1 for q in range(len(meta_f[b]))]
                if sum(own_mask) < 3:
                    continue
                own_c = vecs_f[b][own_mask].mean(0)
                best, best_s = None, -2
                for b2 in batches_of_f:
                    c = own_c if b2 == b else vecs_f[b2].mean(0)
                    s = cos(vecs_f[b][k], c)
                    if s > best_s:
                        best_s, best = s, b2
                correct += (best == b)
                total += 1

        acc = correct / total if total else 0
        chance = 1 / len(batches_of_f) if batches_of_f else 0
        results[variant]["fable5"] = {"accuracy": acc, "chance": chance, "n": total}
        print(f"    {'Fable 5':>10}: {acc:.3f}  (chance {chance:.3f}, n={total})")

    # ================================================================
    # Figure: grouped bars (raw vs residual)
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(ALL_MODELS))
    width = 0.32

    raw_accs = [results["raw"][m]["accuracy"] for m in ALL_MODELS]
    res_accs = [results["residual"][m]["accuracy"] for m in ALL_MODELS]
    chance_vals = [results["raw"][m]["chance"] for m in ALL_MODELS]

    bars1 = ax.bar(x - width/2, raw_accs, width, color=[MODEL_COLORS[m] for m in ALL_MODELS],
                   alpha=0.5, edgecolor="white", linewidth=1.5, label="Raw embedding")
    bars2 = ax.bar(x + width/2, res_accs, width, color=[MODEL_COLORS[m] for m in ALL_MODELS],
                   alpha=0.9, edgecolor="white", linewidth=1.5, label="Position-residualized")

    ax.axhline(chance_vals[0], color="#999", linewidth=1.5, linestyle="--", alpha=0.6,
              label=f"Chance ({chance_vals[0]:.3f})")

    for i in range(len(ALL_MODELS)):
        ax.text(i - width/2, raw_accs[i] + 0.01, f"{raw_accs[i]:.3f}",
               ha="center", fontsize=9, fontweight="bold", alpha=0.7)
        ax.text(i + width/2, res_accs[i] + 0.01, f"{res_accs[i]:.3f}",
               ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in ALL_MODELS], fontsize=13, fontweight="bold")
    ax.set_ylabel("Nearest-centroid accuracy", fontsize=12)
    ax.set_title("Batch Handwriting: Can We Tell Which Notebook an Entry Came From?\n"
                 "Position-residualized = after removing the average arc shape",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(raw_accs), max(res_accs)) * 1.15)
    ax.grid(alpha=0.1, axis="y")

    plt.tight_layout()
    json.dump({v: {m: {k: float(x) for k, x in r.items()} for m, r in res.items()} for v, res in results.items()},
              open(os.path.join(a2_common.out_dir(VERSION), "figure13_handwriting.json"), "w"), indent=2)
    out = os.path.join(a2_common.out_dir(VERSION), "paper_fig_handwriting.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
