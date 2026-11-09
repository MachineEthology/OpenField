"""Diagnostic (helper E, 3 Nov 2026): the two diminutive scripts run unchanged on the June folder itself.

Texture/01_corpus is Texture/Batch_1..15 moved whole on 13 July (00_ADMIN/move_log_2026-07-13.md), so it is
the folder the June scripts globbed. It holds three Proton Drive conflict copies (partial notebooks: 4.8 B12,
7 entries; 4.7 B14, 3 entries; 4.7 B15, 7 entries) that the scripts' glob 'schrodingers_box_opus4*_Batch*.json'
picks up as extra batch-files. This check runs the copied functions on that folder (read-only), with and
without the conflict copies, to see which reproduces the June numbers. Not a V1 producer.
"""
import glob
import os
import sys

sys.argv = [sys.argv[0], "v0"]
import diminutive_curve_v1 as dc  # noqa: E402
import self_referential_diminutive_v1 as sr  # noqa: E402

JUNE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\01_corpus"
files = glob.glob(os.path.join(JUNE, "Batch_*", "schrodingers_box_opus4*_Batch*.json"))
print("files matched by the June glob in 01_corpus:", len(files))
for f in files:
    if "#" in f:
        print("  conflict copy:", os.path.relpath(f, JUNE))

print("\n######## June folder, as globbed (conflict copies included)")
dc.BASE = JUNE
dc.diminutive_curve()
sr.BASE = JUNE
sr.main()

print("\n######## June folder, conflict copies excluded")
_glob = glob.glob
glob.glob = lambda p: [f for f in _glob(p) if "#" not in f]
dc.diminutive_curve()
sr.main()
glob.glob = _glob
