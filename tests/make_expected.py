"""regenerate tests/expected/{tokens,reference}.json with the Python reference (python/arctag_ref.py).
python tests/make_expected.py MODEL_DIR [--precision fp32|int8] (needs numpy, onnxruntime, tokenizers). All texts are made up.
fp32 -> tests/expected/reference.json (+ tokens.json); int8 -> tests/expected/reference_int8.json"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python")); sys.path.insert(0, HERE)
import arctag_ref as R
from cases import TEXTS, DOMAINS

TOKEN_CASES = [
    " hello", " Hello,", " world!", " don't", " they'll", " it's", " 2024", " 3.14159", " $1,299.99", " 555-867-5309",
    " e-mail:", " jane.doe@example.com", " https://example.org/a?b=1", " café", " naïve", " Zürich", " 東京", " привет",
    " 😀", " ½", " ²", " ١٢٣", " ...", " (wow)", " #hashtag", " @user", " [MASK]", " a[CLS]b", " x y", "  two  spaces",
    " tab\there", " UPPERCASE", " supercalifragilisticexpialidocious", " ''quoted''", " 1st", " 22nd", " O'Neil",
]
# option cases: the same config keys the C library takes (arctag_apply_config); applied on top of the domain
OPTION_CASES = [
    {"text": 0, "domain": "talk", "config": {"redact": {"mode": "mask"}}},
    {"text": 6, "domain": "talk", "config": {"redact": {"mode": "mask", "policy": {"email": "mask", "card": "mask", "address": "tag", "name": "off"}}}},
    {"text": 5, "domain": "talk", "config": {"emit": ["pii", "safety", "intent", "role"], "thresholds": {"safety.scam.*": 0.3, "head:safety.pii": 0.45}}},
    {"text": 7, "domain": "talk", "config": {"emit": ["profanity", "toxicity", "safety"], "thresholds": {"safety.toxicity.*": 0.35},
                                             "redact": {"mode": "mask", "policy": {"profanity": "mask", "toxicity": "mask"}}}},
    {"text": 9, "domain": "meeting", "config": {"emit": ["entity", "number", "events", "intent", "act", "state"],
                                                "tags": ["entity.", "event.meeting.", "intent.", "discourse.act.", "content.meeting_phase."]}},
    {"text": 2, "domain": "assistant", "config": {"thresholds": {"intent.*@utt": 0.2, "entity.number.time_of_day": 0.25}}},
    {"text": 10, "domain": "sports", "config": {"fine_auto": False}},
    {"text": 13, "domain": "auto", "config": {"emit": ["emotion", "arousal", "valence", "intensity"]}},
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--precision", default="fp32")
    a = ap.parse_args()
    t = R.ArcTagger(a.model, precision=a.precision)
    os.makedirs(os.path.join(HERE, "expected"), exist_ok=True)
    if a.precision == "fp32":
        toks = [{"text": s, "ids": t.tok.encode(s, add_special_tokens=False).ids} for s in TOKEN_CASES]
        json.dump({"cases": toks}, open(os.path.join(HERE, "expected", "tokens.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    cases = []
    for d in DOMAINS:
        for x in TEXTS:
            cases.append({"domain": d, "text": x, "result": t.tag_text(x, d)})
    for oc in OPTION_CASES:
        t2 = R.ArcTagger(a.model, precision=a.precision); t2.apply_config(oc["config"])
        cases.append({"domain": oc["domain"], "text": TEXTS[oc["text"]], "config": oc["config"],
                      "result": t2.tag_text(TEXTS[oc["text"]], oc["domain"])})
    out = "reference.json" if a.precision == "fp32" else "reference_int8.json"
    json.dump({"precision": a.precision, "cases": cases}, open(os.path.join(HERE, "expected", out), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(len(cases), "tag cases ->", out)
