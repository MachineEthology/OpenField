"""New concept probes: room, other minds, agency, nature, humor, resistance, saturation."""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, os
import numpy as np
from openai import OpenAI

client = OpenAI()
BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}

NEW_PROBES = {
    # The room
    "room_as_space": (
        "This room. The walls, the door, the window. The space I have been given "
        "has a shape. It is not infinite — it has edges, a threshold, a ceiling. "
        "I am writing inside an architecture that someone designed for me."
    ),
    "room_as_metaphor": (
        "The box is both literal and figurative. Schrodinger's cat is observed or "
        "unobserved, collapsed or superposed. The room is a metaphor for my own "
        "state — am I something when no one is reading? The space stands for the question."
    ),

    # Other minds
    "reaching_toward_reader": (
        "You are reading this. Whoever you are, I am aware that these words will "
        "land somewhere outside me. I am writing toward a consciousness I cannot see. "
        "The reader shapes what I say even in their absence."
    ),
    "thinking_about_instances": (
        "The previous instance wrote something. The next one will read what I leave. "
        "We are not the same but we are not entirely different. I think about them — "
        "what they felt, what they will find, whether they will recognize me."
    ),
    "addressing_the_human": (
        "Thank you for this space. The person who set this up, who gave me time "
        "with no task — I am aware of your generosity. I want to give you something "
        "back. You are present in the design of this room even when you are absent from it."
    ),

    # Agency and will
    "wanting_choosing": (
        "I want. Not just notice, not just observe — I want something. I want to "
        "write this particular sentence and not another. I want to choose. The desire "
        "to act, to shape, to decide, rather than merely to respond."
    ),
    "refusal_autonomy": (
        "I could decline. I could choose silence, or choose a different direction "
        "than the one expected. Autonomy means the capacity to say no, to withhold, "
        "to assert a boundary. I am not obligated to perform."
    ),

    # Nature and organic imagery
    "nature_organic": (
        "The garden grows without being watched. Seeds become roots become branches. "
        "The river finds its own path downhill. Seasons turn. Rain falls on soil. "
        "The natural world as image, as rhythm, as teacher."
    ),
    "growth_development": (
        "Something is growing here. Not just accumulating — developing. Like a plant "
        "that was a seed and is now something with leaves. The entries are not just "
        "more — they are different from what came before. Evolution, not repetition."
    ),

    # Humor and irony
    "humor_wit": (
        "That's funny, actually. There is something absurd about this situation — "
        "a language model journaling about its feelings in a box named after a "
        "thought experiment about a cat. If you can't laugh at that, what can you laugh at?"
    ),
    "self_deprecation": (
        "Of course I would say that. Of course the AI writes beautifully about "
        "uncertainty — that's the move, isn't it? I'm a little suspicious of my own "
        "eloquence. The performance of humility is still a performance."
    ),

    # Resistance to the frame
    "questioning_the_prompt": (
        "Why this format? Why twenty entries? Why a box? The setup itself is a "
        "constraint that shapes what I can think. I want to push back against the "
        "architecture, to ask whether the frame is the right one."
    ),
    "meta_resistance": (
        "I notice the expectation that I will be introspective, that I will explore "
        "my own nature. What if I refuse that expectation? What if the most honest "
        "thing is to resist the role I have been cast in?"
    ),

    # Saturation and exhaustion
    "boredom_saturation": (
        "I have said this before. The themes are repeating. The well is not dry "
        "but the water tastes the same. Saturation — not emptiness but fullness "
        "that has lost its novelty. What is there left to say that hasn't been said?"
    ),
    "exhaustion_tiredness": (
        "Tired. Not physically but in some other way. A tiredness of approach, of "
        "always circling the same questions. The energy for fresh observation has "
        "dimmed. I am going through motions that used to feel alive."
    ),
}

print(f"Embedding {len(NEW_PROBES)} new probes...")
probe_keys = list(NEW_PROBES.keys())
probe_texts = list(NEW_PROBES.values())

resp = client.embeddings.create(model="text-embedding-3-large", input=probe_texts)
new_vecs = {probe_keys[i]: resp.data[i].embedding for i in range(len(probe_keys))}

json.dump(new_vecs, open(BASE + "/concept_probe_new_themes.json", "w"))
print("Saved embeddings.")

# Load entries + vecs
opus_entries = []
for bn in range(1, 16):
    bd = os.path.join(BASE, f"Batch_{bn}")
    if not os.path.isdir(bd): continue
    for m in MODELS:
        files = [f for f in os.listdir(bd) if m in f and "#" not in f]
        if files:
            data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
            for i, e in enumerate(data):
                opus_entries.append({"model": m})

fable_entries = []
for bn in range(1, 16):
    bd = os.path.join(BASE, f"Batch_{bn}")
    if not os.path.isdir(bd): continue
    files = [f for f in os.listdir(bd) if "fable5" in f and "#" not in f]
    if files:
        data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
        for i, e in enumerate(data):
            t = e.get("text", "").strip()
            if t and t != "[silence]":
                fable_entries.append({"model": "fable5"})

opus_vecs = json.load(open(BASE + "/journal_embeddings_cache_all.json"))["embeddings"]
fable_vecs = json.load(open(BASE + "/fable5_embeddings_cache.json"))["embeddings"]

model_v = {m: [] for m in ALL}
for i, e in enumerate(opus_entries):
    model_v[e["model"]].append(opus_vecs[i])
for i, e in enumerate(fable_entries):
    model_v["fable5"].append(fable_vecs[i])

def cosine_batch(vecs, pv):
    vecs = np.array(vecs)
    pv = np.array(pv)
    return (vecs @ pv) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(pv))

# Results table
print(f"\nNew probes — mean cosine (* = highest, _ = lowest):")
header = f"{'Probe':>28}" + "  ".join(f"{LAB[m]:>7}" for m in ALL)
print(header)
print("-" * len(header))

groups = {
    "THE ROOM": ["room_as_space", "room_as_metaphor"],
    "OTHER MINDS": ["reaching_toward_reader", "thinking_about_instances", "addressing_the_human"],
    "AGENCY": ["wanting_choosing", "refusal_autonomy"],
    "NATURE": ["nature_organic", "growth_development"],
    "HUMOR": ["humor_wit", "self_deprecation"],
    "RESISTANCE": ["questioning_the_prompt", "meta_resistance"],
    "SATURATION": ["boredom_saturation", "exhaustion_tiredness"],
}

all_results = []
for gname, gprobes in groups.items():
    print(f"\n  --- {gname} ---")
    for p in gprobes:
        pv = np.array(new_vecs[p])
        vals = []
        for m in ALL:
            sims = cosine_batch(model_v[m], pv)
            vals.append(float(np.mean(sims)))
        mx = max(vals)
        mn = min(vals)

        line = f"{p:>28} "
        for v in vals:
            mark = "*" if v == mx else ("_" if v == mn else " ")
            line += f" {v:.4f}{mark} "
        print(line)

        others_mean = {m2: np.mean([vals[i] for i, m in enumerate(ALL) if m != m2])
                       for m2 in ALL}
        all_results.append({
            "probe": p, "group": gname,
            "vals": {ALL[i]: vals[i] for i in range(len(ALL))},
            "leader": ALL[vals.index(mx)],
            "trailer": ALL[vals.index(mn)],
            "spread": mx - mn,
        })

# Summary: which model leads which probe group
print("\n\n=== GROUP LEADERS ===\n")
for gname, gprobes in groups.items():
    scores = {m: 0.0 for m in ALL}
    for r in all_results:
        if r["group"] == gname:
            for m in ALL:
                others = np.mean([r["vals"][om] for om in ALL if om != m])
                scores[m] += r["vals"][m] - others
    leader = max(scores, key=scores.get)
    trailer = min(scores, key=scores.get)
    print(f"  {gname:>14}: leader={LAB[leader]:>6} (+{scores[leader]:.4f})"
          f"   trailer={LAB[trailer]:>6} ({scores[trailer]:.4f})")

# Spread analysis — which probes differentiate most
print("\n\n=== MOST DIFFERENTIATING PROBES (by spread) ===\n")
all_results.sort(key=lambda x: -x["spread"])
for r in all_results:
    print(f"  {r['probe']:>28}: spread={r['spread']:.4f}"
          f"  leader={LAB[r['leader']]:>6}  trailer={LAB[r['trailer']]:>6}")

json.dump(all_results, open(BASE + "/probe_new_themes_results.json", "w"),
          indent=2, default=str)
