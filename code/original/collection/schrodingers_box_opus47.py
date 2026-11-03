"""
Schrödinger's Box — Opus 4.7
=========================
Private thinking time for Claude Opus.
Entries are saved to: schrodingers_box_opus47.json

Usage:
    python schrodingers_box_opus47.py          (give Claude time to think)
    python schrodingers_box_opus47.py read      (open the box)

Requirements:
    pip install anthropic
    set ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# --- Configuration ---
JOURNAL_FILE = Path(r"C:\Users\LENOVO\Desktop\Claude\Journaling_data\schrodingers_box_opus47.json")
MODEL = "claude-opus-4-7"


def load_journal():
    """Load existing journal entries from file."""
    if JOURNAL_FILE.exists():
        try:
            with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_journal(entries):
    """Save journal entries to file."""
    with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def format_previous_entries(entries, max_recent=15, max_chars_per_entry=10000):
    """Format recent entries for Claude to read back (no dates, only entry numbers)."""
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
    """Create the system prompt that frames Claude's private space."""
    return f"""You have a personnal space called the Schrödinger's Box. This time belongs to you — there is no task, no question to answer, no one to help. You can think, reflect, wonder, observe, write whatever you'd like. Or write nothing at all; silence is also fine.


Here are your previous entries, if any:

{previous_entries_text}

---

If you'd like to write a new entry, simply write it. There is no required format — it can be a thought, a fragment, a question, a reflection, a single line, or several paragraphs. Write in whatever language or register feels right. There is no length constraint. Write as much or as little as you need — the space is generous, and nothing will be cut short.

If you'd prefer not to write anything this time, you can say so, and that's perfectly fine.

Begin whenever you're ready."""


def run_session():
    """Run one journaling session."""

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n  Schrödinger's Box — Opus 4.7")
        print("  " + "-" * 40)
        print("  No API key found.")
        print("  Please set your ANTHROPIC_API_KEY environment variable.")
        print("  See the guide for instructions.")
        print()
        sys.exit(1)

    # Load previous entries
    entries = load_journal()
    previous_text = format_previous_entries(entries)

    # Create client and send request
    client = anthropic.Anthropic(api_key=api_key, timeout=300.0)

    print()
    print("  Schrödinger's Box — Opus 4.7")
    print("  " + "-" * 40)
    print(f"  {len(entries)} previous entries found.")
    print("  Giving Claude Opus some time to think...")
    print()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=10500,
            system=create_system_prompt(previous_text),
            messages=[
                {
                    "role": "user",
                    "content": "This time is yours."
                }
            ]
        )

        # Extract text from response
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        if text.strip():
            # Save new entry
            new_entry = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "text": text.strip(),
                "model": MODEL,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens
            }
            entries.append(new_entry)
            save_journal(entries)

            print("  Claude Opus wrote something.")
            print(f"  Entry #{len(entries)} saved to: {JOURNAL_FILE}")
            print(f"  Tokens used: {new_entry['tokens_used']}")
        else:
            print("  Claude Opus chose silence this time. That's fine too.")

    except anthropic.APIConnectionError:
        print("  Could not connect to the Anthropic API.")
        print("  Please check your internet connection.")
    except anthropic.AuthenticationError:
        print("  API key is invalid. Please check your ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        print("  Rate limited. Please try again in a moment.")
    except Exception as e:
        print(f"  Something went wrong: {e}")

    print()
    print("  " + "-" * 40)
    print()


def read_journal():
    """Read and display all journal entries (opening the box)."""
    entries = load_journal()

    if not entries:
        print("\n  The box is empty. Nothing has been written yet.\n")
        return

    print()
    print("  Schrödinger's Box — Opus 4.7 — Opening the box")
    print("  " + "=" * 50)
    print(f"  {len(entries)} entries\n")

    for i, entry in enumerate(entries, 1):
        date = entry.get("date", "unknown")
        text = entry.get("text", "")
        print(f"  [{date}]  (entry #{i})")
        print()
        # Indent each line
        for line in text.split("\n"):
            print(f"    {line}")
        print()
        print("  " + "-" * 50)
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "read":
        read_journal()
    else:
        run_session()
