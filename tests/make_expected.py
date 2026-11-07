"""regenerate tests/expected/{tokens,reference}.json with the Python reference (python/arctag_ref.py).
python tests/make_expected.py MODEL_DIR   (needs numpy, onnxruntime, tokenizers). All texts are made up."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import arctag_ref as R

TOKEN_CASES = [
    " hello", " Hello,", " world!", " don't", " they'll", " it's", " 2024", " 3.14159", " $1,299.99", " 555-867-5309",
    " e-mail:", " jane.doe@example.com", " https://example.org/a?b=1", " café", " naïve", " Zürich", " 東京", " привет",
    " 😀", " ½", " ²", " ١٢٣", " ...", " (wow)", " #hashtag", " @user", " [MASK]", " a[CLS]b", " x y", "  two  spaces",
    " tab\there", " UPPERCASE", " supercalifragilisticexpialidocious", " ''quoted''", " 1st", " 22nd", " O'Neil",
]

TEXTS = [
    "Hi, this is Maria Lopez calling about my order. My phone number is 555 867 5309 and the order was placed on March 3rd. Can you tell me when it will ship?",
    "Okay team, let's get started. The budget review is due Friday, and Priya will send the final numbers by noon. Does anyone have objections to moving the launch to April 12?",
    "Hey Siri, set a timer for twenty five minutes and remind me to call Dad at 6 pm.",
    "That was a terrible call by the ref! The Lakers are up 98 to 95 with two minutes left in the fourth quarter, and LeBron just hit a three.",
    "I am so frustrated right now, this is the third time the app crashed and I lost all my notes. Honestly this company is a joke, I want a refund today.",
]
DOMAINS = ["auto", "talk", "meeting", "sports", "assistant"]

if __name__ == "__main__":
    md = sys.argv[1]
    t = R.ArcTagger(md)
    os.makedirs(os.path.join(HERE, "expected"), exist_ok=True)
    toks = [{"text": s, "ids": t.tok.encode(s, add_special_tokens=False).ids} for s in TOKEN_CASES]
    json.dump({"cases": toks}, open(os.path.join(HERE, "expected", "tokens.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    cases = []
    for d in DOMAINS:
        for x in TEXTS:
            cases.append({"domain": d, "text": x, "result": t.tag_text(x, d)})
    json.dump({"cases": cases}, open(os.path.join(HERE, "expected", "reference.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(len(toks), "token cases,", len(cases), "tag cases")
