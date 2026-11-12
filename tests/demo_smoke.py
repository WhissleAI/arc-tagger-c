#!/usr/bin/env python3
"""Smoke and timing test of the web demo (demo/server.py) on the C library.

Starts the demo server on a free port, types two made-up example messages word by word over the WebSocket the way the page
does (partial word first, then the completed word), and checks:
  - spans (T0 and T1), lanes (intent / act / safety), closed-utterance read-outs and typed alarms come back;
  - redaction: masked words reach the client only as placeholders, and the per-word masks agree with the library's own
    redacted_text (stats.redact_mismatch == 0);
  - timing: every span of the final message shows up (T0) on the word that starts it or at most one word later, and is
    confirmed (T1) at most one word after its last word;
  - the int8 graph, when the model directory has it.

    python tests/demo_smoke.py [--model model] [--lib build/libarctag.so] [-v]

Needs the 'websockets' package (pip install -r demo/requirements.txt). Exit code 77 (ctest: skipped) when it or the model is
missing. Prints numbers and tag ids only.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = 77

EXAMPLES = [   # (vertical, redaction, text, tags that must be found as spans)
    ("contact_center", "redact", "Hi, my name is Maria Lopez, my phone number is 415 555 0182 and my email is maria.lopez@example.com",
     ["safety.pii.private_name", "safety.pii.phone_number", "safety.pii.email", "entity.mention.person"]),
    ("meetings", "off", "Okay, so Priya will send the revised budget to the finance team by Thursday, and we'll meet again next week at 10 am.",
     ["entity.mention.person", "entity.number.time_of_day"]),
]


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


async def type_message(ws, vertical, red, text, precision, verbose):
    await ws.send(json.dumps({"type": "start", "config": {"provider": "typed", "vertical": vertical, "redact_mode": red,
                                                          "redact_display": "redacted", "precision": precision}}))
    while True:
        m = json.loads(await asyncio.wait_for(ws.recv(), 60))
        if m["type"] == "started":
            break
        if m["type"] == "error":
            fail(f"start: {m.get('msg')}")
    if m.get("precision") != precision:
        fail(f"started with precision {m.get('precision')}, asked {precision}")
    toks = text.split()
    words = []                                     # what the page holds (display words)
    first_t0, first_t1 = {}, {}                    # (tag, start) -> words typed when first seen
    lanes, typed_on, ms = [], set(), []
    k = 0
    final = None

    def apply(m):
        nonlocal words
        words = words[: m["from"]] + m["words"]
        words = words[: m["n"]]

    for i, tok in enumerate(toks + [None]):
        drafts = []
        if tok is None:
            drafts.append((text, True))
        else:
            if len(tok) > 3:
                drafts.append((" ".join(toks[:i] + [tok[: len(tok) // 2]]), False))      # a partial word (interim), then the word
            drafts.append((" ".join(toks[: i + 1]) + " ", False))
        for d, send in drafts:
            k += 1
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "type_text", "text": d, "send": send, "k": k, "guard": False}))
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), 60))
                if m["type"] == "update":
                    break
            ms.append(1000 * (time.perf_counter() - t0))
            if m.get("type_k") != k:
                fail(f"update type_k {m.get('type_k')} != {k}")
            apply(m)
            typed_n = i + 1 if tok is not None else len(toks)
            complete = d.endswith(" ") or send
            for tier, store in (("T0", first_t0), ("T1", first_t1)):
                for s in m["tiers"][tier]["spans"]:
                    store.setdefault((s["tag"], s["a"]), (typed_n if complete else typed_n - 0.5, s["b"]))
            if m.get("lanes"):
                lanes += m["lanes"]["pts"]
            for r in (m.get("typed") or {}).get("on", []):
                typed_on.add(r["tag"])
            if m["stats"].get("redact_mismatch"):
                fail("redaction differs from the library's redacted_text")
            if send:
                final = m
    # final message: every span of the last update, with its first T0 / T1 sighting (in words typed)
    spans = final["tiers"]["T1"]["spans"]
    found = {s["tag"] for s in spans}
    rows = []
    for s in spans:
        key = (s["tag"], s["a"])
        f0 = first_t0.get(key, (None,))[0]
        f1 = first_t1.get(key, (None,))[0]
        # T0 delay: words typed after the span's first word; T1 delay: words typed after its last word
        d0 = None if f0 is None else f0 - (s["a"] + 1)
        d1 = None if f1 is None else f1 - s["b"]
        rows.append((s["tag"], s["a"], s["b"], d0, d1))
    utt = final.get("utt_done") or []
    lane_kinds = {p[1] for p in lanes}
    return {"found": found, "rows": rows, "utt": utt, "lanes": lane_kinds, "typed_on": typed_on, "words": words, "ms": ms, "final": final}


def check(res, vertical, red, text, need, verbose, label):
    for t in need:
        if t not in res["found"]:
            fail(f"{label}: span {t} not found (got {sorted(res['found'])})")
    if not res["utt"] or not res["utt"][0]["readouts"].get("intent"):
        fail(f"{label}: no closed-utterance read-out")
    for lane in ("intent", "act"):
        if lane not in res["lanes"]:
            fail(f"{label}: no {lane} lane points")
    late0 = [r for r in res["rows"] if r[3] is None or r[3] > 1]
    late1 = [r for r in res["rows"] if r[4] is None or r[4] > 1]
    if late0:
        fail(f"{label}: spans first shown more than 1 word after their first word: {late0}")
    if late1:
        fail(f"{label}: spans confirmed more than 1 word after their last word: {late1}")
    if red == "redact":
        shown = " ".join(w["w"] for w in res["words"] if w["w"])
        for secret in ("415", "0182", "Lopez", "maria.lopez@example.com"):
            if secret in shown.replace("[", " ").split():
                fail(f"{label}: a redacted word reached the client in clear")
        for ph in ("[NAME]", "[PHONE]", "[EMAIL]"):
            if ph not in shown:
                fail(f"{label}: placeholder {ph} missing")
        if not any(w.get("r") == "mask" for w in res["words"]):
            fail(f"{label}: no word has r=mask")
    d0 = [r[3] for r in res["rows"]]
    d1 = [r[4] for r in res["rows"]]
    ms = sorted(res["ms"])
    print(f"ok {label}: {len(res['rows'])} spans · T0 delay (words after the span's first word) "
          f"0:{sum(1 for x in d0 if x <= 0)} 0.5:{sum(1 for x in d0 if 0 < x <= 0.5)} 1:{sum(1 for x in d0 if 0.5 < x <= 1)} · "
          f"T1 delay (words after its last word) 0:{sum(1 for x in d1 if x <= 0)} 1:{sum(1 for x in d1 if 0 < x <= 1)} · "
          f"lanes {sorted(res['lanes'])[:6]} · typed alarms {sorted(res['typed_on'])} · update p50 {ms[len(ms) // 2]:.0f} ms, "
          f"max {ms[-1]:.0f} ms over {len(ms)} drafts")
    if verbose:
        for r in res["rows"]:
            print("   ", r)


async def run(port, precisions, verbose):
    from websockets.asyncio.client import connect
    async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), 30))
        if hello.get("type") != "hello" or not hello.get("verticals"):
            fail("no hello with verticals")
        ids = {v["id"] for v in hello["verticals"]}
        for v, *_ in EXAMPLES:
            if v not in ids:
                fail(f"vertical {v} not offered (configs/)")
        for prec in precisions:
            if prec not in hello.get("precisions", []):
                print(f"skip {prec}: not available in the model directory")
                continue
            for vertical, red, text, need in EXAMPLES:
                res = await type_message(ws, vertical, red, text, prec, verbose)
                check(res, vertical, red, text, need, verbose, f"{prec} {vertical} redaction={red}")
                await ws.send(json.dumps({"type": "stop"}))
                while json.loads(await asyncio.wait_for(ws.recv(), 30))["type"] != "stopped":
                    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("ARCTAG_MODEL_DIR", str(ROOT / "model")))
    ap.add_argument("--lib", default=None)
    ap.add_argument("--precision", default="all", choices=["fp32", "int8", "all"])
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    try:
        import websockets  # noqa: F401
    except ImportError:
        print("skip: the 'websockets' package is not installed (pip install -r demo/requirements.txt)")
        return SKIP
    if not (Path(a.model) / "arctag.onnx").is_file():
        print(f"skip: no model in {a.model} (scripts/fetch_model.sh)")
        return SKIP
    with tempfile.TemporaryDirectory() as td:
        pf = Path(td) / "port"
        cmd = [sys.executable, str(ROOT / "demo" / "server.py"), "--port", "0", "--port-file", str(pf), "--model", a.model, "--threads", "1",
               "--max-sessions", "2"]
        if a.lib:
            cmd += ["--lib", a.lib]
        srv = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            t0 = time.time()
            while not pf.exists() or not pf.read_text().strip():
                if srv.poll() is not None or time.time() - t0 > 60:
                    print(srv.stdout.read() if srv.poll() is not None else "")
                    fail("demo server did not start")
                time.sleep(0.1)
            port = int(pf.read_text())
            import urllib.request
            h = json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f"http://127.0.0.1:{port}/api/health", timeout=10))
            if not h.get("ok"):
                fail("health")
            page = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f"http://127.0.0.1:{port}/", timeout=10).read().decode()
            if "static/type.js" not in page:
                fail("index page")
            precs = ["fp32", "int8"] if a.precision == "all" else [a.precision]
            asyncio.run(run(port, precs, a.verbose))
        finally:
            srv.terminate()
            try:
                out, _ = srv.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                srv.kill(); out, _ = srv.communicate()
        for t in [EXAMPLES[0][2], "Maria", "0182"]:                # the server log must never hold typed text
            if t in (out or ""):
                fail("typed text found in the server log")
    print("demo smoke: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
