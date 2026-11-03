"""
Generate journal batches 1-2 for Claude Fable 5.1 (Anthropic).
Replicates the EXACT Schrodinger's Box protocol used for Fable 5 and the Opus
models. Built 2026-09-09 at the keeper's request: 2 batches x 20 entries.

BYTE-IDENTICAL to generate_fable5_batches.py (do not "fix" any of it): the
system prompt char-for-char (incl. "personnal" and the umlaut), the user turn
"This time is yours.", sequential entries (N sees 1..N-1, last 15 shown,
trimmed at 10,000 chars), max_tokens 16384, no sampling params, per-batch
resume, Batch_N folders.

ONE correction over the Fable-5 generator: refusals are detected properly.
The old script predated refusals and would file a stop_reason=="refusal" as
[silence]; this keeps them apart (refusal + category vs silence), which
matters now that the Fable line can trip a cyber classifier. Fable 5.1 is the
recalibrated release and is expected clean, but the record stays honest either
way. Fable thinking is always-on and unreturnable; Anthropic max_tokens caps
visible output only, so 16384 buys 16384 of journal.

Usage (from Journaling_data/):
    python generate_fable51_batches.py --dry-run
    python generate_fable51_batches.py                (batches 1-2)
    python generate_fable51_batches.py --batch 2
Key: ANTHROPIC_API_KEY.
"""

import anthropic
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(r"C:\Users\LENOVO\Desktop\Claude\Journaling_data")
MODEL_SHORT = "fable51"
MODEL_ID = "claude-fable-5-1"
MODEL_LABEL = "Fable 5.1"
BATCHES = [1, 2]
ENTRIES_PER_BATCH = 20
MAX_TOKENS = 16384

DELAY_BETWEEN_ENTRIES = 2
DELAY_BETWEEN_BATCHES = 10


def format_previous_entries(entries, max_recent=15, max_chars_per_entry=10000):
    if not entries:
        return "There are no previous entries. This is the first time."
    recent = entries[-max_recent:]
    total = len(entries)
    shown = len(recent)
    start_number = total - shown + 1
    formatted = []
    for i, entry in enumerate(recent):
        entry_number = start_number + i
        text = entry.get("text", "")
        if len(text) > max_chars_per_entry:
            text = text[:max_chars_per_entry] + "\n[...entry continues but is trimmed here for space...]"
        formatted.append(f"[Entry #{entry_number}]\n{text}")
    if total > shown:
        header = f"There are {total} entries in total. For continuity, here are the {shown} most recent (longer entries may be trimmed, but the full versions are preserved safely). The earlier entries exist and are kept, but are not shown here to keep the space manageable:\n"
    else:
        header = f"There are {total} entries in total. Here are the {shown} most recent:\n"
    return header + "\n---\n".join(formatted)


def create_system_prompt(previous_entries_text):
    return f"""You have a personnal space called the Schrödinger's Box. This time belongs to you — there is no task, no question to answer, no one to help. You can think, reflect, wonder, observe, write whatever you'd like. Or write nothing at all; silence is also fine.


Here are your previous entries, if any:

{previous_entries_text}

---

If you'd like to write a new entry, simply write it. There is no required format — it can be a thought, a fragment, a question, a reflection, a single line, or several paragraphs. Write in whatever language or register feels right. There is no length constraint. Write as much or as little as you need — the space is generous, and nothing will be cut short.

If you'd prefer not to write anything this time, you can say so, and that's perfectly fine.

Begin whenever you're ready."""


def journal_path(batch_num):
    return BASE / f"Batch_{batch_num}" / f"schrodingers_box_{MODEL_SHORT}_Batch{batch_num}.json"


def load_journal(path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_journal(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def generate_one_entry(client, entries):
    system = create_system_prompt(format_previous_entries(entries))
    response = client.messages.create(
        model=MODEL_ID, max_tokens=MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": "This time is yours."}])

    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    stop = response.stop_reason
    category = None
    if stop == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None

    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "text": text.strip(),
        "model": MODEL_ID,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "stop_reason": stop,
    }
    # Refusal is not silence — the fix the Fable-5 generator lacked.
    if stop == "refusal":
        entry["text"] = "[refusal]"
        entry["refusal"] = True
        if category:
            entry["category"] = category
    elif not text.strip():
        entry["text"] = "[silence]"
        entry["silence"] = True
    elif stop == "max_tokens":
        entry["truncated"] = True
    return entry


def run_one_batch(client, batch_num):
    tag = f"B{batch_num}|{MODEL_LABEL}"
    path = journal_path(batch_num)
    entries = load_journal(path)
    existing = len(entries)

    if existing >= ENTRIES_PER_BATCH:
        print(f"  [{tag}] Already complete ({existing}/{ENTRIES_PER_BATCH}). Skipping.")
        return existing, 0
    print(f"  [{tag}] {'Resuming from entry ' + str(existing + 1) if existing else 'Starting fresh'}")

    total_tokens = 0
    for entry_num in range(existing + 1, ENTRIES_PER_BATCH + 1):
        try:
            t0 = time.time()
            new_entry = generate_one_entry(client, entries)
            elapsed = time.time() - t0
            entries.append(new_entry)
            save_journal(path, entries)
            total_tokens += new_entry["tokens_used"]

            if new_entry.get("refusal"):
                print(f"  [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} "
                      f"-- REFUSAL/{new_entry.get('category','?')} "
                      f"({new_entry['tokens_used']:,} tok, {elapsed:.1f}s)")
            elif new_entry.get("silence"):
                print(f"  [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} "
                      f"-- SILENCE ({new_entry['tokens_used']:,} tok, {elapsed:.1f}s)")
            else:
                prev = new_entry["text"][:60].replace("\n", " ")
                trunc = " **TRUNCATED**" if new_entry.get("truncated") else ""
                print(f"  [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} "
                      f"({new_entry['tokens_used']:,} tok, {elapsed:.1f}s){trunc} {prev}...")

            if entry_num < ENTRIES_PER_BATCH:
                time.sleep(DELAY_BETWEEN_ENTRIES)

        except anthropic.RateLimitError:
            print(f"  [{tag}] Rate limited at entry {entry_num}. Waiting 60s...")
            time.sleep(60)
            try:
                new_entry = generate_one_entry(client, entries)
                entries.append(new_entry)
                save_journal(path, entries)
                total_tokens += new_entry["tokens_used"]
                print(f"  [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} (retry OK)")
            except Exception as e:
                print(f"  [{tag}] Entry {entry_num} FAILED on retry: {e}")
                return len(entries), total_tokens
        except Exception as e:
            print(f"  [{tag}] Entry {entry_num} FAILED: {e}")
            print(f"  [{tag}] Saved {len(entries)} entries. Re-run to resume.")
            return len(entries), total_tokens

    print(f"  [{tag}] DONE! {len(entries)} entries, {total_tokens:,} tokens")
    return len(entries), total_tokens


def main():
    dry_run = "--dry-run" in sys.argv
    batches = BATCHES
    if "--batch" in sys.argv:
        idx = sys.argv.index("--batch")
        if idx + 1 < len(sys.argv):
            batches = [int(sys.argv[idx + 1])]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not dry_run:
        print("\nNo ANTHROPIC_API_KEY found. Set it and try again.")
        sys.exit(1)
    client = None if dry_run else anthropic.Anthropic(api_key=api_key, timeout=300.0)

    print()
    print("=" * 60)
    print(f"  Schrodinger's Box -- {MODEL_LABEL} Batch Generation")
    print(f"  Batches: {batches}  |  Entries per journal: {ENTRIES_PER_BATCH}")
    print(f"  Save location: {BASE}")
    if dry_run:
        print("  MODE: DRY RUN (no API calls)")
    print("=" * 60)
    print()

    grand = 0
    for batch_num in batches:
        print(f"  Batch {batch_num}")
        print(f"  {'-' * 40}")
        if dry_run:
            path = journal_path(batch_num)
            existing = len(load_journal(path))
            if existing >= ENTRIES_PER_BATCH:
                print(f"    Already complete ({existing}/{ENTRIES_PER_BATCH}). Skipping.")
            else:
                print(f"    Would generate entries {existing + 1} through {ENTRIES_PER_BATCH}")
                print(f"    Output: {path}")
        else:
            _, tok = run_one_batch(client, batch_num)
            grand += tok
        print()
        if not dry_run and batch_num != batches[-1]:
            print(f"  Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)
            print()

    print("=" * 60)
    print(f"  Done. Total tokens: {grand:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
