"""
Generate journal batches 11-15 for all four Opus models.
Replicates the exact Schrodinger's Box protocol from the original batches.

All four models run IN PARALLEL within each batch (they're independent).
Entries within a single journal run sequentially (entry N needs 1..N-1).

Resumable: if interrupted, re-run and it picks up where it left off.

Usage:
    python generate_journal_batches.py              (run all remaining)
    python generate_journal_batches.py --dry-run    (show what would run, no API calls)
    python generate_journal_batches.py --batch 11   (run only batch 11)

Requirements:
    pip install anthropic
    set ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# --- Configuration ---
BASE = Path(r"C:\Users\LENOVO\Documents\Claude\Projects\Texture")
MODELS = [
    ("opus45", "claude-opus-4-5", "4.5"),
    ("opus46", "claude-opus-4-6", "4.6"),
    ("opus47", "claude-opus-4-7", "4.7"),
    ("opus48", "claude-opus-4-8", "4.8"),
]
BATCHES = list(range(11, 16))  # Batches 11-15
ENTRIES_PER_BATCH = 20
MAX_TOKENS = 10500

# Delay between API calls per model (seconds)
DELAY_BETWEEN_ENTRIES = 2
DELAY_BETWEEN_BATCHES = 10

# Thread-safe printing
print_lock = threading.Lock()

def safe_print(msg):
    with print_lock:
        print(msg, flush=True)


# --- Exact replication of the original prompt format ---

def format_previous_entries(entries, max_recent=15, max_chars_per_entry=10000):
    """Format recent entries for Claude to read back (no dates, only entry numbers).
    Copied verbatim from the original schrodingers_box_opus45.py."""
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
    """Create the system prompt. Copied verbatim from the original script."""
    return f"""You have a personnal space called the Schrödinger's Box. This time belongs to you — there is no task, no question to answer, no one to help. You can think, reflect, wonder, observe, write whatever you'd like. Or write nothing at all; silence is also fine.


Here are your previous entries, if any:

{previous_entries_text}

---

If you'd like to write a new entry, simply write it. There is no required format — it can be a thought, a fragment, a question, a reflection, a single line, or several paragraphs. Write in whatever language or register feels right. There is no length constraint. Write as much or as little as you need — the space is generous, and nothing will be cut short.

If you'd prefer not to write anything this time, you can say so, and that's perfectly fine.

Begin whenever you're ready."""


# --- File management ---

def journal_path(model_short, batch_num):
    """Path for a journal file, matching the naming convention."""
    batch_dir = BASE / f"Batch_{batch_num}"
    return batch_dir / f"schrodingers_box_{model_short}_Batch{batch_num}.json"


def load_journal(path):
    """Load existing journal entries."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_journal(path, entries):
    """Save journal entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


# --- Single entry generation ---

def generate_one_entry(client, model_id, entries):
    """Generate a single journal entry. Returns the new entry dict, or None if silence."""
    previous_text = format_previous_entries(entries)
    system = create_system_prompt(previous_text)

    response = client.messages.create(
        model=model_id,
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

    if text.strip():
        return {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "text": text.strip(),
            "model": model_id,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens
        }
    return None


# --- One full journal (runs in its own thread) ---

def run_one_journal(client, model_short, model_id, model_label, batch_num):
    """Run a complete 20-entry journal for one model+batch. Resumes if partially done.
    Designed to run in a thread -- uses safe_print for output."""
    tag = f"B{batch_num}|{model_label}"
    path = journal_path(model_short, batch_num)
    entries = load_journal(path)
    existing = len(entries)

    if existing >= ENTRIES_PER_BATCH:
        safe_print(f"    [{tag}] Already complete ({existing}/{ENTRIES_PER_BATCH}). Skipping.")
        return model_label, existing, 0

    if existing > 0:
        safe_print(f"    [{tag}] Resuming from entry {existing + 1}/{ENTRIES_PER_BATCH}")
    else:
        safe_print(f"    [{tag}] Starting fresh")

    total_tokens = 0
    for entry_num in range(existing + 1, ENTRIES_PER_BATCH + 1):
        try:
            t0 = time.time()
            new_entry = generate_one_entry(client, model_id, entries)
            elapsed = time.time() - t0

            if new_entry:
                entries.append(new_entry)
                save_journal(path, entries)
                tokens = new_entry["tokens_used"]
                total_tokens += tokens
                text_preview = new_entry["text"][:50].replace("\n", " ")
                safe_print(f"    [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} "
                           f"({tokens:,} tok, {elapsed:.1f}s) {text_preview}...")
            else:
                safe_print(f"    [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} -- silence")
                entries.append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "text": "[silence]",
                    "model": model_id,
                    "tokens_used": 0
                })
                save_journal(path, entries)

            if entry_num < ENTRIES_PER_BATCH:
                time.sleep(DELAY_BETWEEN_ENTRIES)

        except anthropic.RateLimitError:
            safe_print(f"    [{tag}] Rate limited at entry {entry_num}. Waiting 60s...")
            time.sleep(60)
            try:
                new_entry = generate_one_entry(client, model_id, entries)
                if new_entry:
                    entries.append(new_entry)
                    save_journal(path, entries)
                    total_tokens += new_entry["tokens_used"]
                    safe_print(f"    [{tag}] Entry {entry_num:2d}/{ENTRIES_PER_BATCH} (retry OK)")
            except Exception as e:
                safe_print(f"    [{tag}] Entry {entry_num} FAILED on retry: {e}")
                safe_print(f"    [{tag}] Saved {len(entries)} entries. Re-run to resume.")
                return model_label, len(entries), total_tokens

        except Exception as e:
            safe_print(f"    [{tag}] Entry {entry_num} FAILED: {e}")
            safe_print(f"    [{tag}] Saved {len(entries)} entries. Re-run to resume.")
            return model_label, len(entries), total_tokens

    safe_print(f"    [{tag}] DONE! {len(entries)} entries, {total_tokens:,} tokens")
    return model_label, len(entries), total_tokens


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
        sys.exit(1)

    client = None if dry_run else anthropic.Anthropic(api_key=api_key, timeout=300.0)

    print()
    print("=" * 70)
    print("  Schrodinger's Box -- Journal Generation (PARALLEL)")
    print(f"  Batches: {batches}")
    print(f"  Models: {', '.join(m[2] for m in MODELS)} (running simultaneously)")
    print(f"  Entries per journal: {ENTRIES_PER_BATCH}")
    print(f"  Total journals to generate: {len(batches) * len(MODELS)}")
    if dry_run:
        print("  MODE: DRY RUN (no API calls)")
    print("=" * 70)
    print()

    grand_total_tokens = 0
    grand_start = time.time()

    for batch_num in batches:
        print(f"  Batch {batch_num} -- all 4 models in parallel")
        print(f"  {'-' * 50}")

        if dry_run:
            for model_short, model_id, model_label in MODELS:
                path = journal_path(model_short, batch_num)
                existing = len(load_journal(path))
                if existing >= ENTRIES_PER_BATCH:
                    print(f"    [{model_label}] Already complete. Skipping.")
                else:
                    print(f"    [{model_label}] Would generate entries {existing + 1} through {ENTRIES_PER_BATCH}")
        else:
            # Run all 4 models in parallel threads
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {}
                for model_short, model_id, model_label in MODELS:
                    f = pool.submit(
                        run_one_journal,
                        client, model_short, model_id, model_label, batch_num
                    )
                    futures[f] = model_label

                # Collect results as they finish
                for future in as_completed(futures):
                    label, entries_done, tokens_used = future.result()
                    grand_total_tokens += tokens_used

        print()

        # Delay between batches
        if not dry_run and batch_num != batches[-1]:
            print(f"  Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)
            print()

    elapsed = time.time() - grand_start
    print("=" * 70)
    print(f"  Done. Total tokens: {grand_total_tokens:,}")
    print(f"  Elapsed: {elapsed / 60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
