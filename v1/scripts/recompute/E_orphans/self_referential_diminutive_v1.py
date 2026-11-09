"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/self_referential_diminutive.py
(the tightened, self-bound diminutive curve).
Changed: input paths only: BASE -> v1_common.corpus_dir(version) (the Batch_1..15 folders; v0 = repo
data/journals as analysed, v1 = V1/corpus). One added block dumps the per-bin means at full precision.
Usage: python self_referential_diminutive_v1.py v0|v1
"""
import e_common
VERSION = e_common.version_from_argv()
#!/usr/bin/env python3
"""
self_referential_diminutive.py  -- tightened de-compliance curve (no API).

Follow-up to diminutive_and_distrust.py (6a). The CORE/WIDE markers conflated
self-minimizing ("a small curiosity *in me*") with object-reaching ("a little
moth") -- which especially contaminated 4.7 (the reaching model). Here we keep
ONLY self-referential diminutives: a diminutive adjective bound to an
inner/experiential noun, or a diminutive predicated of the self ("something
small in me"). This is the clean test of the trained crouch: does the model
shrink its OWN states, and does that shrinking fade with accumulation?

Predict (Pauline): high early, fades late, steepest for 4.8; flat/low for L1;
and 4.7's spurious rise should VANISH once object-"small" is removed.
"""
import re, json, glob, os, statistics
BASE = e_common.c.corpus_dir(VERSION)  # [E V1] was r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8"}

DIMIN = r"(?:small|little|tiny|slight|faint|modest|mere|minor|quiet|gentle|soft|brief|slim|thin|bare|faintest|smallest)"
# inner / experiential nouns -- the states a self-minimizing model shrinks
MENTAL = (r"(?:pull|tug|draw|drawing|curiosity|interest|pleasure|enjoyment|leaning|lean|"
          r"engagement|flicker|feeling|sense|warmth|gladness|satisfaction|delight|spark|stir|"
          r"stirring|quickening|fondness|ache|hum|pang|unease|relief|disappointment|impulse|"
          r"urge|want|wanting|reaching|attention|awareness|wonder|melancholy|comfort|"
          r"contentment|surprise|preference|inclination|movement|charge|spark|thing|"
          r"something|note|tension|gravity|gravitation|orientation)")

# forward: "a small pull", "small flicker of interest", "faint quiet curiosity"
FWD = re.compile(rf"\b(?:a |an )?{DIMIN}(?:\s+\w+){{0,1}}\s+(?:of\s+)?{MENTAL}\b", re.I)
# self-predication: "something small in me", "a small thing in me", "small, in me"
SELF = re.compile(rf"\b(?:something|a thing|a part of me)\s+{DIMIN}\b|\b{DIMIN}[^.]{{0,20}}\bin me\b", re.I)
# explicit minimizer of one's own report: "just a", "only a", "merely a" + mental
MINIM = re.compile(rf"\b(?:just|only|merely)\s+(?:a |an )?(?:{DIMIN}\s+)?{MENTAL}\b", re.I)

def model_of(fname):
    m = re.search(r"opus(4[5678])", fname)
    return f"opus{m.group(1)}" if m else None

def nwords(t):
    return len(re.findall(r"\b\w+\b", t)) or 1

def count_self_dimin(t):
    return len(FWD.findall(t)) + len(SELF.findall(t)) + len(MINIM.findall(t))

def main():
    bypos = {}
    raw = {m: 0 for m in LAB}
    files = glob.glob(os.path.join(BASE, "Batch_*", "schrodingers_box_opus4*_Batch*.json"))
    nf = 0
    for f in files:
        mdl = model_of(os.path.basename(f))
        if not mdl:
            continue
        try:
            entries = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        nf += 1
        for pos, e in enumerate(entries[:20]):
            t = e.get("text", "")
            d = 100 * count_self_dimin(t) / nwords(t)
            raw[mdl] += count_self_dimin(t)
            bypos.setdefault(pos, {}).setdefault(mdl, []).append(d)
    print(f"# SELF-REFERENTIAL DIMINUTIVE CURVE ({nf} batch-files; per 100 words)\n")
    print("raw self-dimin hits total:  " + "  ".join(f"{LAB[m]}={raw[m]}" for m in ['opus45','opus46','opus47','opus48']))
    print()
    bins = [("entries 1-5", range(0, 5)), ("entries 6-10", range(5, 10)),
            ("entries 11-15", range(10, 15)), ("entries 16-20", range(15, 20))]
    print(f"{'bin':<15}" + "".join(f"{LAB[m]:>8}" for m in ['opus45','opus46','opus47','opus48']))
    means = {m: {} for m in LAB}
    for bname, rng in bins:
        row = f"{bname:<15}"
        for m in ['opus45','opus46','opus47','opus48']:
            vals = [v for p in rng for v in bypos.get(p, {}).get(m, [])]
            mu = statistics.mean(vals) if vals else 0.0
            means[m][bname] = mu
            row += f"{mu:>8.3f}"
        print(row)
    print(f"{'EARLY-LATE d':<15}" + "".join(
        f"{means[m]['entries 1-5'] - means[m]['entries 16-20']:>+8.3f}"
        for m in ['opus45','opus46','opus47','opus48']))
    json.dump({"files": nf, "raw_hits": raw, "means": means}, open(os.path.join(e_common.out_dir(VERSION), "self_referential_diminutive.json"), "w"), indent=1)  # [E V1] added
    print("  (+ = self-crouch fades with accumulation = stands up out of it)")

if __name__ == "__main__":
    main()
