"""Conformance checks for arctag-serve: every route, the error contract, realtime streaming, and parity with the Python
reference (python/arctag_ref.py: numpy + onnxruntime + tokenizers) on the same text.

    python deploy/conformance/conformance.py --url http://127.0.0.1:8080 --model-dir model \
        [--python /path/to/python-with-onnxruntime] [--skip-reference] [--report out.json]

Parity: the reference tags tests/cases.py's 16 texts in 7 domains (112 cases) plus examples/data/*.txt; the server's
POST /v1/tags output must have the same structure, the same tags, word ranges, texts and on / off decisions, and scores
within --tol (default 5e-4: the native backend's documented tolerance against the reference, see README "Parity").
Streaming: words pushed over WS /v1/realtime must give the same spans as one offline call.

Standard library only (the reference run needs numpy, onnxruntime==1.20.1 and tokenizers).
Part of arc-tagger-c (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import http_client as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
results = []


def check(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def compare(a, b, tol, path="", out=None, stats=None):
    """structural compare: same keys / lengths / strings / bools / ints, floats within tol"""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a) - {"model"}, set(b) - {"model"}
        if ka != kb:
            out.append(f"{path}: keys {sorted(ka ^ kb)}")
        for k in ka & kb:
            compare(a[k], b[k], tol, f"{path}.{k}", out, stats)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} vs {len(b)} items")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, tol, f"{path}[{i}]", out, stats)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None or b is None:
        if a != b:
            out.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = abs(a - b)
        if stats is not None:
            stats["max"] = max(stats.get("max", 0.0), d)
        if d > tol:
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


REF_SCRIPT = r'''
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "python"))
import arctag_ref as R
model, cases_path, out = sys.argv[2], sys.argv[3], sys.argv[4]
t = R.ArcTagger(model, precision=os.environ.get("ARCTAG_PRECISION", "fp32"))
res = []
for c in json.load(open(cases_path)):
    res.append(t.tag_text(c["text"], c["domain"]))
json.dump(res, open(out, "w"))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model-dir", help="the model folder, for the Python reference")
    ap.add_argument("--python", default=sys.executable, help="interpreter with numpy, onnxruntime, tokenizers")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--report")
    a = ap.parse_args()
    url = a.url
    text = "Hi, this is Maria Lopez calling about my order. My phone number is 555 867 5309. Can you tell me when it will ship?"

    # ---- health and metadata
    for p in ("/health", "/v1/health/live", "/v1/health/ready", "/ping"):
        r = H.get(url, p)
        check(f"GET {p}", r.status == 200, f"{r.status} {r.text.strip()}")
    r = H.get(url, "/v1/models")
    m = r.json() if r.status == 200 else {}
    ids = [x["id"] for x in m.get("data", [])]
    check("GET /v1/models", r.status == 200 and ids, ",".join(ids))
    check("GET /v1/models/{id}", bool(ids) and H.get(url, f"/v1/models/{ids[0]}").status == 200)
    check("GET /execution-parameters (SageMaker batch)", H.get(url, "/execution-parameters").status == 200)

    # ---- POST /v1/tags
    r = H.post_json(url, "/v1/tags", {"text": text, "domain": "talk"})
    o = r.json() if r.status == 200 else {}
    types = {s["type"] for s in o.get("spans", [])}
    check("POST /v1/tags {text, domain}", r.status == 200 and "safety.pii.phone_number" in types and o.get("utterances"),
          f"{len(o.get('spans', []))} spans, {len(o.get('utterances', []))} utterances")
    r = H.request(url, "POST", "/v1/tags?domain=talk", text.encode(), {"Content-Type": "text/plain"})
    check("POST /v1/tags text/plain", r.status == 200 and r.json().get("profile", "").startswith("talk"))
    words = [{"w": w, "t0": i * 0.3, "t1": i * 0.3 + 0.25, "conf": 0.9, "spk": "A"} for i, w in enumerate(text.split())]
    r = H.post_json(url, "/v1/tags", {"words": words, "domain": "talk"})
    check("POST /v1/tags {words: [{w, t0, t1, conf, spk}]}", r.status == 200 and r.json().get("n_words") == len(words))
    segs = [{"text": "Hi, this is Maria Lopez.", "start": 0.0, "end": 1.8, "speaker": "agent"},
            {"text": "My number is 555 867 5309.", "start": 2.0, "end": 4.0, "speaker": "customer",
             "words": [{"word": w, "start": 2.0 + 0.4 * i, "end": 2.35 + 0.4 * i} for i, w in enumerate("My number is 555 867 5309.".split())]}]
    r = H.post_json(url, "/v1/tags", {"segments": segs, "domain": "talk"})
    check("POST /v1/tags {segments} (transcript in, e.g. Arc Transcriber verbose_json)", r.status == 200 and len(r.json().get("utterances", [])) == 2,
          f"{len(r.json().get('utterances', []))} utterances" if r.status == 200 else r.text[:120])
    r = H.post_json(url, "/v1/tags", {"inputs": [text, {"text": "The Lakers are up 98 to 95.", "domain": "nba"}], "domain": "talk"})
    res = r.json().get("results", []) if r.status == 200 else []
    check("POST /v1/tags batch {inputs: [...]}", r.status == 200 and len(res) == 2 and res[1].get("profile", "").startswith("sports"),
          ", ".join(x.get("profile", "?") for x in res))
    r = H.post_json(url, "/v1/tags", {"text": "Email me at jo@example.com or call 415 555 0199.", "redact": "mask", "emit": "pii"})
    check("per-request options (redact, emit)", r.status == 200 and "[EMAIL]" in r.json().get("redacted_text", ""), r.json().get("redacted_text", r.text)[:80])
    r = H.post_json(url, "/v1/tags", {"text": "Email me at jo@example.com."})
    check("options do not leak into the next request", r.status == 200 and "redacted_text" not in r.json())

    # ---- errors
    check("invalid JSON -> 400", H.request(url, "POST", "/v1/tags", b"{nope", {"Content-Type": "application/json"}).status == 400)
    check("no text / words / segments -> 400", H.post_json(url, "/v1/tags", {"foo": 1}).status == 400)
    check("load-time option per request -> 400", H.post_json(url, "/v1/tags", {"text": "hi", "precision": "int8"}).status == 400)
    check("unknown domain -> 400", H.post_json(url, "/v1/tags", {"text": "hi", "domain": "martian"}).status == 400)
    check("audio body -> 415", H.request(url, "POST", "/v1/tags", b"RIFF....WAVE", {"Content-Type": "audio/wav"}).status == 415)
    check("unknown route -> 404", H.get(url, "/v2/x").status == 404)
    check("GET on a POST route -> 405", H.get(url, "/v1/tags").status == 405)

    # ---- SageMaker /invocations
    r = H.post_json(url, "/invocations", {"text": text})
    check("POST /invocations JSON", r.status == 200 and r.json().get("spans"))
    r = H.post_json(url, "/invocations", {"instances": [{"text": text}, {"text": "Set a timer for ten minutes.", "domain": "assistant"}]})
    check("POST /invocations JSON batch (instances)", r.status == 200 and len(r.json().get("results", [])) == 2)
    body = (json.dumps({"text": text}) + "\n" + json.dumps({"text": "Objection, Your Honor.", "domain": "court"}) + "\n").encode()
    r = H.request(url, "POST", "/invocations", body, {"Content-Type": "application/jsonlines"})
    check("POST /invocations application/jsonlines", r.status == 200 and len(r.text.strip().splitlines()) == 2)

    # ---- Snowflake service function shape
    r = H.post_json(url, "/v1/snowflake/tags", {"data": [[0, text], [1, "The Lakers are up 98 to 95.", "nba"], [2, "Email jo@example.com", "", '{"redact": "mask"}']]})
    d = r.json().get("data", []) if r.status == 200 else []
    check("POST /v1/snowflake/tags (SPCS service function rows)", r.status == 200 and [x[0] for x in d] == [0, 1, 2]
          and d[1][1].get("profile", "").startswith("sports") and "[EMAIL]" in d[2][1].get("redacted_text", ""))

    # ---- realtime: words streamed one at a time == one offline pass
    ws = H.WS(url, "/v1/realtime?domain=talk")
    first = ws.recv_json()
    check("WS /v1/realtime: session.created", first and first.get("type") == "session.created")
    spans = {}
    utts = 0
    tw = text.split()
    for i, w in enumerate(tw):
        ws.send_json({"type": "words", "words": [w], "endpoint": w.endswith((".", "?", "!"))})
        u = ws.recv_json()
        if u.get("type") != "tags.update":
            check("WS /v1/realtime: tags.update", False, json.dumps(u)[:200])
            break
        a0 = u["a0"]
        spans = {k: v for k, v in spans.items() if k[1] <= a0}
        for s in u["spans"]:
            spans[(s["type"], s["end"], s["start"])] = s
        utts += len(u["utterances"])
    ws.send_json({"type": "session.close"})
    ws.close()
    off = H.post_json(url, "/v1/tags", {"text": text, "domain": "talk"}).json()
    so = sorted((s["type"], s["start"], s["end"]) for s in off["spans"])
    ss = sorted((t, st, en) for (t, en, st) in spans)
    check("WS /v1/realtime: streamed spans == offline spans", so == ss, f"{len(ss)} spans, {utts} utterances")
    ws = H.WS(url, "/v1/realtime")
    ws.recv_json()
    ws.send_json({"type": "text", "text": "Call me at 415 555 0199.", "endpoint": True})
    u = ws.recv_json()
    check("WS /v1/realtime: {type: text}", u.get("type") == "tags.update" and len(u.get("utterances", [])) == 1)
    ws.close()

    # ---- concurrency
    codes = []
    def one():
        codes.append(H.post_json(url, "/v1/tags", {"text": text}).status)
    ts = [threading.Thread(target=one) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    check("8 concurrent requests", codes == [200] * 8, str(codes))

    # ---- parity with the Python reference
    if not a.skip_reference:
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        from cases import DOMAINS, TEXTS
        cases = [{"text": t, "domain": d} for d in DOMAINS for t in TEXTS]
        for f in sorted(glob.glob(os.path.join(ROOT, "examples", "data", "*.txt"))):
            t = " ".join(open(f).read().split())
            if t:
                cases.append({"text": " ".join(t.split()[:600]), "domain": "auto"})
        tmp = tempfile.mkdtemp(prefix="arctag-conf-")
        cp, rp, sp = os.path.join(tmp, "cases.json"), os.path.join(tmp, "ref.json"), os.path.join(tmp, "ref.py")
        json.dump(cases, open(cp, "w"))
        open(sp, "w").write(REF_SCRIPT)
        p = subprocess.run(["nice", "-n", "19", a.python, sp, ROOT, a.model_dir, cp, rp], capture_output=True, text=True, timeout=3600)
        if p.returncode:
            check("reference: python/arctag_ref.py", False, p.stderr[-500:])
        else:
            refs = json.load(open(rp))
            bad, stats, n_spans = 0, {}, 0
            first_diffs = []
            for c, ref in zip(cases, refs):
                got = H.post_json(url, "/v1/tags", c).json()
                d = compare(ref, got, a.tol, stats=stats)
                n_spans += len(ref.get("spans", []))
                if d:
                    bad += 1
                    first_diffs.append(f"[{c['domain']}] {c['text'][:40]}...: " + "; ".join(d[:3]))
            check(f"parity vs Python reference ({len(cases)} cases)", bad == 0,
                  f"{n_spans} reference spans, {bad} cases differ, max |score diff| {stats.get('max', 0):.2e} (tol {a.tol:g})")
            for x in first_diffs[:5]:
                print("   ", x)

    n_ok = sum(r["ok"] for r in results)
    print(f"\n{n_ok}/{len(results)} checks passed")
    if a.report:
        json.dump({"url": url, "passed": n_ok, "total": len(results), "results": results}, open(a.report, "w"), indent=1)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
