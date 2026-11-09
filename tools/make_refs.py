#!/usr/bin/env python3
"""Reference outputs from ONNX Runtime, for the native parity check. usage: make_refs.py MODEL_DIR OUT.bin

Cases: the texts of tests/expected/reference.json in their domains, long made-up texts (up to 1024 words), and one text with
timings, speakers, confidences and interim words. Model inputs come from python/arctag_ref.py, so features are checked by
tests/test_parity.c (through the public API) and the forward pass by this file's output (tests/test_native_parity.c).

File: int32 n_cases, int32 n_outputs, n_outputs int32 column counts, then per case: int32 W, int32 dom, int32 arrays pcs [W,8],
npc [W], wfl [W,5], feats [W,8], spk [W], ust [W], then each output as float32 [W, columns]."""
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import arctag_ref as R  # noqa: E402

ORDER = ["frame", "span_beg", "span_in", "span_end", "ev_at", "ev_imm", "ev_rec", "utt", "speaker", "item",
         "sh_entity_number_t", "sh_entity_number_b", "sh_entity_mention_t", "sh_entity_mention_b", "sh_safety_pii_t", "sh_safety_pii_b",
         "sh_safety_profanity_t", "sh_safety_profanity_b", "sh_safety_toxicity_t", "sh_safety_toxicity_b", "ver_m", "ver_e"]


def main():
    md, out = sys.argv[1:3]
    t = R.ArcTagger(md, threads=1)
    ref = json.load(open(os.path.join(ROOT, "tests", "expected", "reference.json")))
    cases = []
    for c in ref["cases"]:
        ws = R.split_words(c["text"])
        cases.append((c["domain"], [{"w": w} for w in ws], R.utt_starts(ws)))
    base = " ".join(c["text"] for c in ref["cases"][:5]).split()
    for n, dom in ((8, "auto"), (32, "meeting"), (150, "auto"), (512, "auto"), (1024, "meeting")):
        ws = (base * (n // len(base) + 1))[:n]
        cases.append((dom, [{"w": w} for w in ws], R.utt_starts(ws)))
    tx = base[:90]
    rich = [{"w": w, "t0": 0.31 * i, "t1": 0.31 * i + 0.25, "conf": 0.5 + 0.01 * (i % 40), "spk": "AB"[(i // 7) % 2], "final": i % 5 != 0} for i, w in enumerate(tx)]
    cases.append(("meeting", rich, R.utt_starts(tx)))
    with open(out, "wb") as f:
        dims = None
        f.write(struct.pack("<i", len(cases)))
        for dom, words, us in cases:
            _, prof = t.profile(dom)
            us = list(us); us[0] = 1
            fe = t.features(words, None)
            O = t.run(fe, us, prof["dom"])
            W = len(words)
            if dims is None:
                dims = [int(np.prod(O[n].shape[1:])) for n in ORDER]
                f.write(struct.pack("<i", len(ORDER)) + struct.pack(f"<{len(ORDER)}i", *dims))
            pcs = np.zeros((W, 8), np.int32); npc = np.zeros(W, np.int32)
            for i, (p, _, _, _) in enumerate(fe):
                pcs[i, :len(p)] = p; npc[i] = len(p)
            wfl = np.array([x[1] for x in fe], np.int32); feats = np.array([x[2] for x in fe], np.int32)
            spk = np.array([x[3] for x in fe], np.int32); ust = np.array(us, np.int32)
            f.write(struct.pack("<ii", W, prof["dom"]))
            for a in (pcs, npc, wfl, feats, spk, ust):
                f.write(a.tobytes())
            for nm in ORDER:
                f.write(O[nm].astype(np.float32).reshape(W, -1).tobytes())
    print(f"references: {len(cases)} cases from ONNX Runtime, longest {max(len(c[1]) for c in cases)} words")


if __name__ == "__main__":
    main()
