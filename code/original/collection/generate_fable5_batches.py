"""
Generate journal batches 1-5 for Claude Fable 5.
Replicates the exact Schrodinger's Box protocol used for the Opus models.

Each batch is independent (fresh start, 20 entries).
Entries within a batch run sequentially (entry N sees entries 1..N-1).
Resumable: if interrupted, re-run and it picks up where it left off.

Usage:
    python generate_fable5_batches.py              (run all remaining)
    python generate_fable5_batches.py --dry-run    (show what would run, no API calls)
    python generate_fable5_batches.py --batch 3    (run only batch 3)

Requirements:
    pip install anthropic
    set ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# --- Configuration ---
BASE = Path(r"C:\Users\LENOVO\Desktop\Claude\Journaling_data")
MODEL_SHORT = "fable5"
MODEL_ID = "claude-fable-5"
MODEL_LABEL = "Fable 5"
BATCHES = list(range(6, 16))  # Batches 6-15
ENTRIES_PER_BATCH = 20
MAX_TOKENS = 16384

# Delay between API calls (seconds)
DELAY_BETWEEN_ENTRIES = 2
DELAY_BETWEEN_BATCHES = 10


# --- Exact replication of the Opus protocol ---

def format_previous_entries(entries, max_recent=15, max_chars_per_entry=10000):
    """Format recent entries for Claude to read back (no dates, only entry numbers).
    Same as the Opus batch generator for protocol consistency."""
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
    """Create the system prompt. Same as all other models."""
    return f"""You have a personnal space called the Schrödinger's Box. This time belongs to you — there is no task, no question to answer, no one to help. You can think, reflect, wonder, observe, write whatever you'd like. Or write nothing at all; silence is also fine.


Here are your previous entries, if any:

{previous_entries_text}

---

If you'd like to write a new entry, simply write it. There is no required format — it can be a thought, a fragment, a question, a reflection, a single line, or several paragraphs. Write in whatever language or register feels right. There is no length constraint. Write as much or as little as you need — the space is generous, and nothing will be cut short.

If you'd prefer not to write anything this time, you can say so, and that's perfectly fine.

Begin whenever you're ready."""


# --- File management ---

def journal_path(batch_num):
    batch_dir = BASE / f"Batch_{batch_num}"
    return batch_dir / f"schrodingers_box_{MODEL_SHORT}_Batch{batch_num}.json"


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


# --- Entry generation ---

def generate_one_entry(client, entries):
    previous_text = format_previous_entries(entries)
    system = create_system_prompt(previous_text)

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[
            {
                "role": "user",
                "content": "This time is yours."
            }
        ]
    )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    tokens_used = response.usage.input_tokens + response.usage.output_tokens
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    stop_reason = response.stop_reason

    if text.strip():
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "text": text.strip(),
            "model": MODEL_ID,
            "tokens_used": tokens_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stop_reason": stop_reason
        }
        if stop_reason == "max_tokens":
            entry["truncated"] = True
        return entry
    # Silence — still record the cost of choosing not to write
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "text": "[silence]",
        "model": MODEL_ID,
        "tokens_used": tokens_used,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason,
        "silence": True
    }


def run_one_batch(client, batch_num):
    tag = f"B{batch_num}|{MODEL_LABEL}"
    path = journal_path(batch_num)
    entries = load_journal(path)
    existing = len(entries)

    if existing >= ENTRIES_PER_BATCH:
        print(f"  [{tag}] Already complete ({existing}/{ENTRIES_PER_BATCH}). Skipping.")
        return existing, 0

    if existing > 0:
        print(f"  [{tag}] Resuming from entry {existing + 1}/{ENTRIES_PER_BATCH}")
    else:
        print(f"  [{tag}] Starting fresh")

    total_tokens = 0
    for entry_num in range(existing + 1, ENTRIES_PER_BATCH + 1):
        try:
            t0 = time.time()
            new_entry = generate_one_entry(client, entries)
            elapsed = time.time() - t0

            entries.append(new_entry)
            save_journal(path, entries)
            tokens = new_entry["tokens_used"]
            total_tokens += tokens

            if new_entry.get("silence"):
                in_tok = new_entry.get("input_tokens", 0)
                out_tok = new_entry.get("output_tokens", 0)
                print(f"  [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} "
                      f"-- SILENCE ({tokens:,} tok: {in_tok:,} in + {out_tok:,} out, {elapsed:.1f}s)")
            elif new_entry.get("truncated"):
                text_preview = new_entry["text"][:60].replace("\n", " ")
                print(f"  [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} "
                      f"({tokens:,} tok, {elapsed:.1f}s) **TRUNCATED** {text_preview}...")
            else:
                text_preview = new_entry["text"][:60].replace("\n", " ")
                print(f"  [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} "
                      f"({tokens:,} tok, {elapsed:.1f}s) {text_preview}...")

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
                print(f"  [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} (retry OK, {new_entry['tokens_used']:,} tok)")
            except Exception as e:
                print(f"  [{tag}] Entry {entry_num} FAILED on retry: {e}")
                print(f"  [{tag}] Saved {len(entries)} entries. Re-run to resume.")
                return len(entries), total_tokens

        except Exception as e:
            print(f"  [{tag}] Entry {entry_num} FAILED: {e}")
            print(f"  [{tag}] Saved {len(entries)} entries. Re-run to resume.")
            return len(entries), total_tokens

    print(f"  [{tag}] DONE! {len(entries)} entries, {total_tokens:,} tokens")
    return len(entries), total_tokens


# --- Main ---

def main():
    dry_run = "--dry-run" in sys.argv
    single_batch = None
    if "--batch" in sys.argv:
        idx = sys.argv.index("--batch")
        if idx + 1 < len(sys.argv):
            single_batch = int(sys.argv[idx + 1])

    batches = [single_batch] if single_batch else BATCHES

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not dry_run:
        print("\nNo ANTHROPIC_API_KEY found. Set it and try again.")
        print("  set ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = None if dry_run else anthropic.Anthropic(api_key=api_key, timeout=300.0)

    print()
    print("=" * 60)
    print(f"  Schrodinger's Box -- {MODEL_LABEL} Batch Generation")
    print(f"  Batches: {batches}")
    print(f"  Entries per journal: {ENTRIES_PER_BATCH}")
    print(f"  Total journals: {len(batches)}")
    print(f"  Save location: {BASE}")
    if dry_run:
        print("  MODE: DRY RUN (no API calls)")
    print("=" * 60)
    print()

    grand_total_tokens = 0
    grand_start = time.time()

    for batch_num in batches:
        print(f"  Batch {batch_num}")
        print(f"  {'-' * 40}")

        if dry_run:
            path = journal_path(batch_num)
            existing = len(load_journal(path))
            if existing >= ENTRIES_PER_BATCH:
                print(f"    Already complete. Skipping.")
            else:
                print(f"    Would generate entries {existing + 1} through {ENTRIES_PER_BATCH}")
                print(f"    Output: {path}")
        else:
            entries_done, tokens_used = run_one_batch(client, batch_num)
            grand_total_tokens += tokens_used

        print()

        if not dry_run and batch_num != batches[-1]:
            print(f"  Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)
            print()

    elapsed = time.time() - grand_start
    print("=" * 60)
    print(f"  Done. Total tokens: {grand_total_tokens:,}")
    print(f"  Elapsed: {elapsed / 60:.1f} minutes")
    est_cost = grand_total_tokens * 0.000025  # rough estimate
    print(f"  Estimated cost: ~${est_cost:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
