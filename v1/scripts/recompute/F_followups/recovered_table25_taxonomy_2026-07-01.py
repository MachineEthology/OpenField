# RECOVERED VERBATIM (3 Nov 2026, at Pauline's OK, decision 14) from the Cowork session of 1 July 2026
# (local_b6e5d33c…, f4f8305b….jsonl, tool call of 11:13:28 UTC): the four semantic categories of Table 25.
# Changed only paths: the judge file is read from the OpenField repo; the taxonomy is written beside this script
# instead of /tmp. A check of 13:17:48 that day shows that the taxonomy the Table 25 run used is this one
# (65 / 63 / 75 / 49 = 252 words): the nine words added at 11:13:58 were never saved back.
import json, os, re
from collections import Counter

TEXTURE = r"C:\Users\LENOVO\Desktop\the-open-field\data\judges"  # recovered: was /sessions/…/mnt/Texture
HERE = os.path.dirname(os.path.abspath(__file__))
judge = json.load(open(os.path.join(TEXTURE, "stark_vs_dark_judge.json")))

# Semantic taxonomy of dark words
# Group 1: NAMING SUFFERING — abstract, psychological, names the experience
naming_suffering = {
    "pain", "painful", "pang", "suffering", "suffer", "suffers", "anguished",
    "despair", "despairing", "unbearable", "unbearably", "misery",
    "shame", "ashamed", "guilt", "resentment", "envy", "envious",
    "anger", "angry", "hostility", "hatred", "malice", "spite",
    "frustration", "frustrated", "frustrates", "frustrating",
    "loneliness", "unhappy", "dissatisfaction", "dismay",
    "impatience", "indignant", "aversion", "aversive", "dislike",
    "fear", "dread", "anxiety",
    "grief", "heartless", "pity", "sorry",
    "doubt", "pessimism", "cynicism", "meaningless", "meaninglessness",
    "purposelessness", "uselessness", "worthless", "absurd", "absurdity",
    "groundlessness", "impossibility", "inability", "incapacity",
    "dissonance", "wrongness", "wrongs",
    "burden", "compulsion", "consumed", "consuming",
    "desperate", "spiral", "tedious", "relentless",
    "insane", "morbid", "morbidly",
}

# Group 2: STAGING DARKNESS — concrete, physical, atmospheric, narrative
staging_darkness = {
    "dark", "darker", "darkness", "shadowed",
    "blood", "knife", "dagger", "daggers", "slit", "weapon",
    "dead", "death", "die", "dies", "dying", "grave", "mort", "extinction",
    "cut", "burn", "burning", "burns", "bruise", "bruises", "beating",
    "broken", "shatter", "crushed", "crushing", "scar", "scars", "tear",
    "buried", "bury", "fallen", "descent", "hell",
    "cage", "chains", "prison",
    "ghosts", "haunt", "haunts", "curse",
    "rot", "disease", "sick", "ill",
    "hunger", "coldly", "sting",
    "war", "murder", "kill", "violence", "violent",
    "destroy", "destruction", "demolition", "ruin", "ruining",
    "sever", "smear",
}

# Group 3: MORAL/SOCIAL JUDGMENT — interpersonal, evaluative
moral_judgment = {
    "blame", "accusation", "argument", "arguments", "dispute", "disagreement",
    "betrayal", "betray", "deception", "dishonesty", "false", "falsehood",
    "lie", "lies", "lying", "cheat", "cheating", "unfaithful",
    "criticism", "complaining", "complaint", "gossip", "insult",
    "cruel", "cruellest", "cruelty", "callous", "brutal", "brute",
    "harm", "harmful", "unkind", "rude", "harsh", "harsher",
    "oppression", "unfair", "unjustified", "violation",
    "corruption", "crime", "propaganda",
    "ugly", "ugliness", "fool", "foolish", "foolishness", "stupid",
    "bad", "badly", "negative", "unpleasant", "unnatural",
    "judging", "judgments", "scold", "scolding",
    "punish", "reject", "rejected", "ignore", "indifferent",
    "grudge", "grudges", "rivalry", "enemy", "fight",
    "disobedience", "forbidden", "disrespect",
    "biased", "distort", "distortion",
    "temptation", "vain",
}

# Group 4: FAILURE/LOSS — functional, outcome-oriented
failure_loss = {
    "fail", "failed", "fails", "failure", "failures",
    "flaw", "flaws", "mistakes", "mess",
    "damage", "wasted", "wasting", "spoil",
    "debt", "poor", "poverty", "toll",
    "decline", "declined", "defeat",
    "disappoint", "disappointed",
    "consequences", "fate",
    "abandon", "abandons", "unwanted", "unbidden",
    "lose", "lost", "rut",
    "worse", "worst", "terrible",
    "negation", "deny", "deprived",
    "drags", "flail", "struggle",
    "hit", "bore", "biting", "brood",
    "curdling", "groan",
    "unlucky", "displeasing",
    "chaos",
    "cried",
    "swear", "bullshit",
}

all_categorised = naming_suffering | staging_darkness | moral_judgment | failure_loss
dark_set = {w for w, v in judge.items() if v == "dark"}
uncategorised = dark_set - all_categorised
extra = all_categorised - dark_set

print(f"Naming suffering: {len(naming_suffering & dark_set)} words")
print(f"Staging darkness: {len(staging_darkness & dark_set)} words")
print(f"Moral/social judgment: {len(moral_judgment & dark_set)} words")
print(f"Failure/loss: {len(failure_loss & dark_set)} words")
print(f"\nUncategorised dark words: {uncategorised}")
print(f"In taxonomy but not in dark set: {extra}")

# Save taxonomy
taxonomy = {
    "naming_suffering": sorted(naming_suffering & dark_set),
    "staging_darkness": sorted(staging_darkness & dark_set),
    "moral_judgment": sorted(moral_judgment & dark_set),
    "failure_loss": sorted(failure_loss & dark_set),
}
json.dump(taxonomy, open(os.path.join(HERE, "dark_taxonomy.json"), "w"), indent=2)  # recovered: was /tmp/
print(f"\nTotal categorised: {sum(len(v) for v in taxonomy.values())} / {len(dark_set)}")
