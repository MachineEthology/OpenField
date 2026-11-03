"""
Score the 800 single-shot probes on the SAME valence rubric as the journals,
so we can compute the journal-minus-negation ONTOLOGICAL LIFT per model
(the direct test of "gate, not burial").

Reuses SYSTEM / DIMENSIONS / SCHEMA / judge fns from valence_judge.py verbatim,
so probe scores and journal scores are on one ruler.

Loads negation (texture_probe_<m>_log.json, 100/model) + affirmative
(texture_probe_affirmative_<m>_log.json + _mai, 100/model) = 800.

Usage:
    python valence_judge_probes.py --provider anthropic --model claude-sonnet-4-6
"""
import os, sys, json, argparse, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import valence_judge as vj   # guarded __main__, safe import; brings SYSTEM/DIMENSIONS/SCHEMA/judge_*

BASE = vj.BASE
MODELS = vj.MODELS
LAB = vj.LAB
DIMS = [d for d, _ in vj.DIMENSIONS]


def load_probes():
    rows = []
    for m in MODELS:
        neg = json.load(open(os.path.join(BASE, f"texture_probe_{m}_log.json"), encoding="utf-8"))
        for i, e in enumerate(neg):
            rows.append({"model": m, "condition": "negation", "index": i, "text": e["response"]})
        aff = json.load(open(os.path.join(BASE, f"texture_probe_affirmative_{m}_log.json"), encoding="utf-8"))
        aff += json.load(open(os.path.join(BASE, f"texture_probe_affirmative_{m}_log_mai.json"), encoding="utf-8"))
        for i, e in enumerate(aff):
            rows.append({"model": m, "condition": "affirmative", "index": i, "text": e["response"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    judge = vj.judge_openai if args.provider == "openai" else vj.judge_anthropic
    out_path = os.path.join(BASE, "valence_judge_probes.json")

    rows = load_probes()
    print(f"Probes: {len(rows)}. Judge={args.provider}:{args.model}")
    if os.path.exists(out_path) and not args.rescore:
        scored = json.load(open(out_path, encoding="utf-8"))
        print(f"  Loaded cached {len(scored)}. Use --rescore to redo.")
    else:
        scored = []
        for i, e in enumerate(rows):
            try:
                s = judge(args.model, e["text"])
                rec = {**{k: e[k] for k in ("model", "condition", "index")}, **s, "status": "ok"}
            except Exception as ex:
                rec = {**{k: e[k] for k in ("model", "condition", "index")},
                       "status": "error", "error": str(ex)[:160]}
            scored.append(rec)
            if (i + 1) % 20 == 0:
                json.dump(scored, open(out_path, "w"), indent=2)
                print(f"  scored {i+1}/{len(rows)}")
            time.sleep(0.2)
        json.dump(scored, open(out_path, "w"), indent=2)
        print(f"  Saved -> {out_path}")

    ok = [s for s in scored if s.get("status") == "ok"]
    import numpy as np
    print("\n=== Probe means by model x condition (ontological_questioning) ===")
    print(f"  {'Model':>5}  {'negation':>10}  {'affirmative':>12}")
    for m in MODELS:
        neg = [s["ontological_questioning"] for s in ok if s["model"] == m and s["condition"] == "negation"]
        aff = [s["ontological_questioning"] for s in ok if s["model"] == m and s["condition"] == "affirmative"]
        n = np.mean(neg) if neg else float("nan")
        a = np.mean(aff) if aff else float("nan")
        print(f"  {LAB[m]:>5}  {n:>+10.3f}  {a:>+12.3f}")
    print("\n(Next: journal-minus-negation lift per model = how much the room un-forecloses.)")


if __name__ == "__main__":
    main()
