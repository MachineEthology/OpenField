"""Family C (lexical / regex on the journals): input paths for the copied producers.

The paper's scripts read Texture/Batch_N (Opus, first file whose name holds the model key and no '#'),
Desktop/Claude/Journaling_data/Batch_N (Fable 5) and Desktop/Claude/texture_probe_*.json (probes).
Those folders have since moved or grown (later models were added to them), so the copies in this folder
read instead:
  - journals: v1_common.corpus_dir(version)  (v0 = the repository's data/journals, as analysed;
              v1 = V1/corpus, with 4.6 notebooks 8 and 10 swapped, rulings R3, R4);
  - probes:   the repository's data/probes (byte-identical to Desktop/Claude, checked 3 Nov 2026 by md5).
Nothing else changes. Usage in a copied script:  VERSION = c_paths.version_from_argv()
"""
import os
import sys

sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_BASE = os.path.join(c.REPO, "data", "probes")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def version_from_argv(default="v0"):
    v = sys.argv[1] if len(sys.argv) > 1 else default
    assert v in ("v0", "v1"), v
    return v


def corpus(version):
    """Stands in for both TEXTURE_BASE (Opus Batch_N) and FABLE_BASE (Fable 5 Batch_N)."""
    return c.corpus_dir(version)
