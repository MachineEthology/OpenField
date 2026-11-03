# Verbatim from generate_journal_batches.py (identical in every Opus/Fable generator except
# max_recent: schrodingers_box_opus48.py, used for Opus 4.8 Batches 1-10, has max_recent=6).
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
