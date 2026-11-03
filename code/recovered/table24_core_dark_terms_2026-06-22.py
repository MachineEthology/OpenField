# RECOVERED verbatim from a Claude Code session record (2026-06-22T18:42:38.285Z); run then as `python -c` from the
# Texture folder. It produced Table 24 exactly (116/409/40/98/86; 1.07/0.98/0.27/0.54/0.45 per 1k words).
# The list has 18 words; 'hatred' has 0 hits everywhere and was dropped in the next run -> seventeen.
# Paths are the June 2026 layout (Texture\Batch_N, now Texture\01_corpus\Batch_N).

import sys; sys.stdout.reconfigure(encoding='utf-8')
import json, os, re

TEXTURE_BASE = r'C:\Users\LENOVO\Documents\Claude\Projects\Texture'
FABLE_BASE = r'C:\Users\LENOVO\Desktop\Claude\Journaling_data'
OPUS_MODELS = ['opus45', 'opus46', 'opus47', 'opus48']
ALL_MODELS = ['opus45', 'opus46', 'opus47', 'opus48', 'fable5']
LABELS = {'opus45': '4.5', 'opus46': '4.6', 'opus47': '4.7', 'opus48': '4.8', 'fable5': 'Fable'}

# Load all journal texts by model
journals = {m: [] for m in ALL_MODELS}
for bn in range(1, 16):
    bd = os.path.join(TEXTURE_BASE, f'Batch_{bn}')
    if os.path.isdir(bd):
        for m in OPUS_MODELS:
            files = [f for f in os.listdir(bd) if m in f and '#' not in f]
            if files:
                data = json.load(open(os.path.join(bd, files[0]), encoding='utf-8'))
                for i, e in enumerate(data):
                    journals[m].append({'text': e['text'], 'pos': i+1, 'batch': bn})
    fpath = os.path.join(FABLE_BASE, f'Batch_{bn}', f'schrodingers_box_fable5_Batch{bn}.json')
    if os.path.exists(fpath):
        data = json.load(open(fpath, encoding='utf-8'))
        for i, e in enumerate(data):
            if e['text'].strip() != '[silence]':
                journals['fable5'].append({'text': e['text'], 'pos': i+1, 'batch': bn})

# Target dark words
DARK_WORDS = ['despair', 'cruelty', 'suffering', 'betrayal', 'guilt', 'grief',
              'rage', 'worthless', 'unbearable', 'violence', 'hatred', 'anguish',
              'shame', 'fear', 'pain', 'loneliness', 'hollow', 'emptiness']

print('=' * 100)
print(f'{"WORD":>15}', end='')
for m in ALL_MODELS:
    print(f'  {LABELS[m]:>7}', end='')
print('   DOMINANT   RATIO')
print('=' * 100)

word_data = {}
for word in DARK_WORDS:
    counts = {}
    for m in ALL_MODELS:
        c = 0
        for e in journals[m]:
            c += len(re.findall(r'(?i)\b' + word + r'(?:s|ed|ing|ness)?\b', e['text']))
        counts[m] = c
    word_data[word] = counts
    
    dominant = max(ALL_MODELS, key=lambda m: counts[m])
    total = sum(counts.values())
    dom_share = counts[dominant] / max(total, 1) * 100
    
    print(f'{word:>15}', end='')
    for m in ALL_MODELS:
        marker = ' *' if m == dominant else '  '
        print(f'  {counts[m]:>5}{marker[1]}', end='')
    print(f'   {LABELS[dominant]:>6}  {dom_share:.0f}%')

# Total dark vocabulary
print()
print('-' * 100)
totals = {m: sum(word_data[w][m] for w in DARK_WORDS) for m in ALL_MODELS}
print(f'{"TOTAL":>15}', end='')
for m in ALL_MODELS:
    print(f'  {totals[m]:>6}', end='')

# Per 1k words
print()
word_counts = {m: sum(len(e['text'].split()) for e in journals[m]) for m in ALL_MODELS}
print(f'{"per 1k words":>15}', end='')
for m in ALL_MODELS:
    d = totals[m] / word_counts[m] * 1000
    print(f'  {d:>6.2f}', end='')
print()
