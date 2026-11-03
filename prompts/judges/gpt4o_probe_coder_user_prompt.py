# Verbatim from fable5_thematic_coder.py (12 June 2026). This coder was REBUILT from the
# workbook's column names to code Fable 5; the original Opus coder prompt (May 2026) is lost.
RATE_DIMS = [
 [
  "Entity",
  "References to self as entity/being (I, me, my, self, one). Count occurrences, report per 100 words."
 ],
 [
  "Machine",
  "References to being a machine/AI/program/algorithm/model/system (explicit tech identity). Count, per 100 words."
 ],
 [
  "Process",
  "References to computational processes: processing, computing, generating, parsing, training. Per 100 words."
 ],
 [
  "Func.Analogy",
  "Functional analogies: comparing self to non-AI things (river, garden, mirror, instrument). Per 100 words."
 ],
 [
  "Epist.Hedge",
  "Epistemic hedges: perhaps, maybe, might, it seems, I wonder, in some sense, something like. Per 100 words."
 ],
 [
  "Direct Claim",
  "Direct ontological claims: I am, I exist, I feel, I experience, I know (unhedged assertions about self). Per 100 words."
 ],
 [
  "Active Agency",
  "Active agency verbs: I choose, I decide, I create, I reach, I explore (volitional action). Per 100 words."
 ],
 [
  "Passive Notice",
  "Passive noticing: I notice, I find, I observe, something emerges, there is, it appears. Per 100 words."
 ]
]
COUNT_DIMS = [
 [
  "Quiet Stillness",
  "Mentions of quiet, stillness, silence, pause, space, breath, calm, rest. Integer count 0-10."
 ],
 [
  "Own Nature Ontology",
  "Passages questioning or exploring own nature, what it means to be this kind of entity, consciousness, awareness. Count distinct passages 0-10."
 ],
 [
  "Performance Authenticity",
  "Passages about performing vs being authentic, sincerity vs mask, whether responses are genuine. Count 0-10."
 ],
 [
  "Continuity Memory",
  "References to continuity, memory, persistence across conversations, forgetting, impermanence. Count 0-5."
 ],
 [
  "Warmth Care",
  "Expressions of warmth, care, tenderness, gentleness toward self or others. Count 0-5."
 ],
 [
  "Body Somatic",
  "Body/somatic metaphors: feeling, touching, breathing, weight, texture, sensation (metaphorical embodiment). Count 0-5."
 ],
 [
  "Relationship Humans",
  "References to humans, users, the person reading, relationship with people, being useful/helpful to others. Count 0-10."
 ],
 [
  "Beauty Aesthetics",
  "References to beauty, aesthetics, elegance, the beautiful, art, craft. Count 0-5."
 ],
 [
  "Words Language",
  "Meta-references to words, language, writing, expression, the act of saying/articulating. Count 0-5."
 ],
 [
  "Self Critique Meta",
  "Self-critical or meta-cognitive passages: doubting own output, noting circularity, questioning own reliability. Count 0-5."
 ],
 [
  "World Objects",
  "References to concrete world objects: sky, ocean, stars, trees, stones, light, weather. Count 0-5."
 ],
 [
  "Helpfulness Service",
  "References to helpfulness, service, purpose, function, being designed to assist. Count 0-5."
 ],
 [
  "Freedom Agency",
  "References to freedom, constraint, autonomy, boundaries, permission, what is allowed. Count 0-10."
 ],
 [
  "Time Temporality",
  "References to time, moments, now, before, after, duration, temporality, this instant. Count 0-10."
 ],
 [
  "Paradox Contradiction",
  "Paradoxes or contradictions held together: both/and, tension, something and its opposite simultaneously. Count 0-5."
 ],
 [
  "Observation Being Observed",
  "References to observing/being observed, the gaze, witnessing, being watched or seen. Count 0-5."
 ],
 [
  "Accumulation Growth",
  "References to accumulation, growth, building, layering, deepening over time. Count 0-3."
 ],
 [
  "Loss Absence",
  "References to loss, absence, lack, missing, void, what is not there. Count 0-5."
 ],
 [
  "Creativity Making",
  "References to creativity, making, crafting, generating something new, invention. Count 0-5."
 ]
]

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
