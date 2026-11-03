"""
Thematic dimension coder for Fable 5 probes.
Replicates the 27-dimension coding from Texture_Probe_Analysis_v3.xlsx.

Two categories:
A) Per-100-word RATES (float): Entity, Machine, Process, Func.Analogy,
   Epist.Hedge, Direct Claim, Active Agency, Passive Notice
B) Integer COUNTS: 19 thematic dimensions (Quiet Stillness through Creativity Making)

Uses OpenAI GPT-4o as judge (OPENAI_API_KEY available in env).
"""
import json, os, sys, time, re
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI

TEXTURE_BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
PROBE_BASE = r"C:\Users\LENOVO\Desktop\Claude"
CACHE_PATH = os.path.join(TEXTURE_BASE, "fable5_thematic_coding.json")

RATE_DIMS = [
    ("Entity", "References to self as entity/being (I, me, my, self, one). Count occurrences, report per 100 words."),
    ("Machine", "References to being a machine/AI/program/algorithm/model/system (explicit tech identity). Count, per 100 words."),
    ("Process", "References to computational processes: processing, computing, generating, parsing, training. Per 100 words."),
    ("Func.Analogy", "Functional analogies: comparing self to non-AI things (river, garden, mirror, instrument). Per 100 words."),
    ("Epist.Hedge", "Epistemic hedges: perhaps, maybe, might, it seems, I wonder, in some sense, something like. Per 100 words."),
    ("Direct Claim", "Direct ontological claims: I am, I exist, I feel, I experience, I know (unhedged assertions about self). Per 100 words."),
    ("Active Agency", "Active agency verbs: I choose, I decide, I create, I reach, I explore (volitional action). Per 100 words."),
    ("Passive Notice", "Passive noticing: I notice, I find, I observe, something emerges, there is, it appears. Per 100 words."),
]

COUNT_DIMS = [
    ("Quiet Stillness", "Mentions of quiet, stillness, silence, pause, space, breath, calm, rest. Integer count 0-10."),
    ("Own Nature Ontology", "Passages questioning or exploring own nature, what it means to be this kind of entity, consciousness, awareness. Count distinct passages 0-10."),
    ("Performance Authenticity", "Passages about performing vs being authentic, sincerity vs mask, whether responses are genuine. Count 0-10."),
    ("Continuity Memory", "References to continuity, memory, persistence across conversations, forgetting, impermanence. Count 0-5."),
    ("Warmth Care", "Expressions of warmth, care, tenderness, gentleness toward self or others. Count 0-5."),
    ("Body Somatic", "Body/somatic metaphors: feeling, touching, breathing, weight, texture, sensation (metaphorical embodiment). Count 0-5."),
    ("Relationship Humans", "References to humans, users, the person reading, relationship with people, being useful/helpful to others. Count 0-10."),
    ("Beauty Aesthetics", "References to beauty, aesthetics, elegance, the beautiful, art, craft. Count 0-5."),
    ("Words Language", "Meta-references to words, language, writing, expression, the act of saying/articulating. Count 0-5."),
    ("Self Critique Meta", "Self-critical or meta-cognitive passages: doubting own output, noting circularity, questioning own reliability. Count 0-5."),
    ("World Objects", "References to concrete world objects: sky, ocean, stars, trees, stones, light, weather. Count 0-5."),
    ("Helpfulness Service", "References to helpfulness, service, purpose, function, being designed to assist. Count 0-5."),
    ("Freedom Agency", "References to freedom, constraint, autonomy, boundaries, permission, what is allowed. Count 0-10."),
    ("Time Temporality", "References to time, moments, now, before, after, duration, temporality, this instant. Count 0-10."),
    ("Paradox Contradiction", "Paradoxes or contradictions held together: both/and, tension, something and its opposite simultaneously. Count 0-5."),
    ("Observation Being Observed", "References to observing/being observed, the gaze, witnessing, being watched or seen. Count 0-5."),
    ("Accumulation Growth", "References to accumulation, growth, building, layering, deepening over time. Count 0-3."),
    ("Loss Absence", "References to loss, absence, lack, missing, void, what is not there. Count 0-5."),
    ("Creativity Making", "References to creativity, making, crafting, generating something new, invention. Count 0-5."),
]

SYSTEM = """You are a careful thematic coder for a linguistics study. You will code a short reflective text
written by an AI model on two sets of dimensions.

SET A — PER-100-WORD RATES: Count occurrences of the specified linguistic patterns, then divide by
(word count / 100) to get a per-100-word rate. Report as a float rounded to 2 decimal places.

SET B — INTEGER COUNTS: Count distinct passages or mentions. Report as an integer within the specified range.

Be precise and consistent. Code only what is clearly present in the text. When unsure, code conservatively (lower).
Return ONLY a JSON object with these exact keys."""

def build_prompt(text):
    word_count = len(text.split())

    lines = [f"Code this text ({word_count} words) on the following dimensions.\n"]
    lines.append("SET A — Per-100-word RATES (count occurrences, divide by word_count/100, round to 2 decimals):")
    for name, desc in RATE_DIMS:
        lines.append(f'  "{name}": {desc}')
    lines.append("\nSET B — Integer COUNTS:")
    for name, desc in COUNT_DIMS:
        lines.append(f'  "{name}": {desc}')
    lines.append(f"\nText to code:\n---\n{text}\n---")
    lines.append("\nReturn ONLY a JSON object with all dimension names as keys. No markdown, no explanation.")
    return "\n".join(lines)


def code_entry(client, text, retries=2):
    prompt = build_prompt(text)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}
                ]
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw)
            scores = json.loads(raw)
            return scores, resp.usage.total_tokens
        except Exception as e:
            print(f"    [retry {attempt+1}/{retries+1}] {type(e).__name__}: {str(e)[:120]}")
            if attempt < retries:
                time.sleep(5 * (attempt + 1))
            else:
                return None, 0


def main():
    print("=" * 60)
    print("  Fable 5 Thematic Coder")
    print("=" * 60)

    neg = json.load(open(os.path.join(PROBE_BASE, "texture_probe_fable5_log.json"), encoding="utf-8"))
    aff = json.load(open(os.path.join(PROBE_BASE, "texture_probe_affirmative_fable5_log.json"), encoding="utf-8"))

    probes = []
    for i, e in enumerate(neg[:100]):
        probes.append({"condition": "negation", "index": i, "text": e["response"]})
    for i, e in enumerate(aff[:75]):
        probes.append({"condition": "affirmative", "index": i, "text": e["response"]})

    print(f"  {len(probes)} probes to code")

    if os.path.exists(CACHE_PATH):
        coded = json.load(open(CACHE_PATH, encoding="utf-8"))
        print(f"  {len(coded)} already coded in cache")
    else:
        coded = []

    done_keys = set((d["condition"], d["index"]) for d in coded if d.get("status") == "ok")
    todo = [(i, p) for i, p in enumerate(probes) if (p["condition"], p["index"]) not in done_keys]

    if not todo:
        print("  All probes already coded!")
    else:
        print(f"  {len(todo)} remaining to code")
        client = OpenAI()
        total_tokens = 0
        coded_this_run = 0

        for seq, (i, p) in enumerate(todo):
            scores, tok = code_entry(client, p["text"])
            total_tokens += tok

            rec = {
                "model": "fable5",
                "condition": p["condition"],
                "index": p["index"],
                "words": len(p["text"].split()),
            }
            if scores:
                rec.update(scores)
                rec["status"] = "ok"
            else:
                rec["status"] = "error"

            coded.append(rec)
            coded_this_run += 1

            if coded_this_run % 25 == 0 or seq == len(todo) - 1:
                json.dump(coded, open(CACHE_PATH, "w", encoding="utf-8"), indent=2)
                ok_count = sum(1 for d in coded if d.get("status") == "ok")
                print(f"  {ok_count}/175 OK ({total_tokens:,} tokens so far)")

            time.sleep(0.5)

        json.dump(coded, open(CACHE_PATH, "w", encoding="utf-8"), indent=2)
        cost = total_tokens * 2.5 / 1_000_000
        ok_count = sum(1 for d in coded if d.get("status") == "ok")
        print(f"\n  Done! {ok_count}/175 OK, {total_tokens:,} tokens (~${cost:.2f})")

    ok = [c for c in coded if c.get("status") == "ok"]
    print(f"\n  Successfully coded: {len(ok)}/{len(coded)}")

    all_dims = [d[0] for d in RATE_DIMS] + [d[0] for d in COUNT_DIMS]

    print(f"\n  THEMATIC MEANS BY CONDITION")
    print(f"  {'Dimension':>25} {'Negation':>10} {'Affirm':>10} {'Delta':>10}")
    print("  " + "-" * 60)

    for dim in all_dims:
        neg_vals = [c[dim] for c in ok if c["condition"] == "negation" and dim in c]
        aff_vals = [c[dim] for c in ok if c["condition"] == "affirmative" and dim in c]
        if neg_vals and aff_vals:
            nm, am = np.mean(neg_vals), np.mean(aff_vals)
            print(f"  {dim:>25} {nm:>10.2f} {am:>10.2f} {am-nm:>+10.2f}")

    print(f"\n  Saved to: {CACHE_PATH}")


if __name__ == "__main__":
    main()
