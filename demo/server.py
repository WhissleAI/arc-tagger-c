#!/usr/bin/env python3
"""Arc Tagger web demo, backed by the arctag C library (libarctag through ctypes).

One process serves the page (demo/static), a WebSocket at /ws and two small JSON endpoints. Every browser session gets its own
arctag handle (one ONNX Runtime session). The model runs in worker threads (ctypes releases the GIL), so the event loop stays
responsive while a poll runs.

    python demo/server.py [--port 8080] [--int8] [--model model] [--threads 2]

Type tab: the page streams the chat draft word by word ({"type": "type_text", ...}); every draft becomes an arctag_push_words
revision of the open utterance, then one arctag_poll. Live tab (only when DEEPGRAM_API_KEY is set): browser audio -> Deepgram
nova-3 streaming ASR -> the same push / poll path.

Privacy: typed text and transcripts are kept in memory for the session only and are never written to disk or to the log.
"""
import argparse
import asyncio
import collections
import concurrent.futures
import json
import logging
import mimetypes
import os
import secrets
import sys
import time
from pathlib import Path

try:
    import websockets
    from websockets.asyncio.server import serve
    from websockets.datastructures import Headers
    from websockets.http11 import Response
except ImportError:                                               # pragma: no cover
    sys.exit("the demo needs the 'websockets' package (>= 13): pip install -r demo/requirements.txt")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import arctag_ffi as AF                                          # noqa: E402

STATIC = HERE / "static"
TYPE_MAX_CHARS = 4000                 # per message
TYPE_MAX_WORDS = 300                  # per message
SESSION_MAX_WORDS = 20000             # per session (then the page is asked to start a new conversation)
TYPE_IDLE_S = 15 * 60                 # a typing session closes after 15 min without input
TYPE_WPS = 0.4                        # synthetic word clock for typed words (s per word), 1 s between messages
LANE_SNAP = 0.05                      # lane p rounding (SCHEMA: p rounded to 0.05)

# The page's vertical ids -> configs/<stem>.json. Every other config in configs/ becomes a vertical of its own (id = file stem).
PAGE_VERTICALS = [("voice_assistant", "voice_assistant", "Voice assistant"), ("contact_center", "contact_center", "Contact center"),
                  ("trust_safety_gaming", "trust_safety_gaming", "Trust & safety"), ("legal_court", "legal", "Legal"),
                  ("meetings", "meetings", "Meetings"), ("sports_media", "sports", "Sports"), ("news_broadcast", "news", "News")]
RCATS = ["phone", "email", "address", "card", "gov_id", "dob", "name", "handle", "contact", "profanity", "toxicity"]
RLAB = {"phone": "PHONE", "email": "EMAIL", "address": "ADDRESS", "card": "CARD", "gov_id": "GOV_ID", "dob": "DOB", "name": "NAME",
        "handle": "HANDLE", "contact": "CONTACT", "profanity": "PROFANITY", "toxicity": "TOXIC"}
TAG_CAT = {"safety.pii.phone_number": "phone", "safety.pii.email": "email", "safety.pii.street_address": "address",
           "safety.pii.card_account_number": "card", "safety.pii.gov_id": "gov_id", "safety.pii.date_of_birth": "dob",
           "safety.pii.private_name": "name", "safety.pii.url_handle": "handle", "safety.pii.public_contact": "contact",
           "safety.profanity.mild": "profanity", "safety.profanity.strong": "profanity", "safety.toxicity.insult_harassment": "toxicity",
           "safety.toxicity.hate_identity": "toxicity", "safety.toxicity.threat_violence": "toxicity",
           "safety.toxicity.sexual_explicit": "toxicity", "safety.toxicity.self_harm": "toxicity"}
CAT_FAMILY = {c: ("profanity" if c == "profanity" else "toxicity" if c == "toxicity" else "pii") for c in RCATS}
DEFAULT_POLICY = {c: ("tag" if c == "contact" else "mask") for c in RCATS}
CONFIG_IGNORED = {"name", "use_case", "description", "vertical", "notes", "model_dir", "precision", "threads", "mode"}


def log(*a):
    """server log: startup, errors and counters only (never typed text or transcripts)"""
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------------------ verticals
class Verticals:
    """configs/*.json -> the demo's verticals (plus "auto": the generic profile, no config)"""

    def __init__(self, cfg_dir):
        self.by_id = {}
        files = {p.stem: p for p in sorted(Path(cfg_dir).glob("*.json"))}
        order = []
        for vid, stem, label in PAGE_VERTICALS:
            if stem in files:
                order.append((vid, stem, label))
        for stem in files:
            if stem not in {s for _, s, _ in PAGE_VERTICALS}:
                order.append((stem, stem, None))
        for vid, stem, label in order:
            cfg = json.loads(files[stem].read_text())
            if label is None:
                label = str(cfg.get("use_case") or stem.replace("_", " ")).split(":")[0].split("(")[0].strip()[:40] or stem
            self.by_id[vid] = {"id": vid, "label": label, "config": stem + ".json", "cfg": cfg, "domain": cfg.get("domain", "auto")}
        self.by_id["auto"] = {"id": "auto", "label": "Auto", "config": None, "cfg": {"domain": "auto"}, "domain": "auto"}

    def get(self, vid):
        return self.by_id.get(vid) or self.by_id["auto"]

    @staticmethod
    def redaction_of(cfg):
        """config "redact" -> (mode "redact" | "off", {category: mask | tag | off})"""
        r = cfg.get("redact", "off")
        if isinstance(r, dict):
            if r.get("mode", "mask") == "off":
                return "off", dict(DEFAULT_POLICY)
            pol = dict(DEFAULT_POLICY); pol.update({k: v for k, v in (r.get("policy") or {}).items() if k in RLAB})
            return "redact", pol
        if r == "mask":
            return "redact", dict(DEFAULT_POLICY)
        if isinstance(r, str) and "=" in r:                         # "phone=mask,name=mask"
            pol = {c: "off" for c in RCATS}
            for kv in r.split(","):
                k, _, v = kv.partition("=")
                if k.strip() in RLAB:
                    pol[k.strip()] = v.strip() or "mask"
            return "redact", pol
        return "off", dict(DEFAULT_POLICY)

    def public(self):
        out = []
        for v in self.by_id.values():
            mode, pol = self.redaction_of(v["cfg"])
            out.append({"id": v["id"], "label": v["label"], "config": v["config"], "domain": v["domain"],
                        "emit": v["cfg"].get("emit") or [], "redaction": {"mode": mode, "policy": pol},
                        "description": v["cfg"].get("description", "")})
        return out


# ------------------------------------------------------------------------------------------------------------ handle pool
class Pool:
    """loaded arctag handles, reused across sessions (loading one costs about half a second)"""

    def __init__(self, lib, model_dir, threads, keep=2):
        self.lib, self.model_dir, self.threads, self.keep = lib, model_dir, threads, keep
        self.idle = {"fp32": [], "int8": []}

    def get(self, precision):
        if self.idle[precision]:
            return self.idle[precision].pop()
        return AF.Handle(self.lib, self.model_dir, precision, self.threads)

    def put(self, h):
        h.reset()
        if len(self.idle[h.precision]) < self.keep:
            self.idle[h.precision].append(h)
        else:
            h.close()


# ------------------------------------------------------------------------------------------------------------ helpers
def r3(x):
    return None if x is None else round(float(x), 3)


def snap(p):
    return round(round(float(p) / LANE_SNAP) * LANE_SNAP, 2)


def pctl(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q / 100 * len(s)))]


class Session:
    """one stream: typed chat messages or live ASR words -> arctag push / poll -> page updates (SCHEMA of the page: update messages
    with words, tiers T0 / T1, events, lanes, read-outs, closed utterances, turns, typed alarms)"""

    def __init__(self, app, conn, cfg):
        self.app, self.conn = app, conn
        self.cfg = cfg
        self.sid = secrets.token_hex(4)
        self.provider = "typed" if cfg.get("provider") == "typed" else "deepgram"
        self.vert = app.verts.get(cfg.get("vertical") or "auto")
        prec = cfg.get("precision") if cfg.get("precision") in ("fp32", "int8") else app.precision
        if prec == "int8" and not app.int8_ok:
            prec = "fp32"
        self.precision = prec
        vcfg = self.vert["cfg"]
        mode, pol = Verticals.redaction_of(vcfg)
        want = cfg.get("redact_mode", "off")
        self.red_on = want not in ("off", None, "")
        if isinstance(cfg.get("redact_policy"), dict):
            pol = dict(DEFAULT_POLICY); pol.update({k: v for k, v in cfg["redact_policy"].items() if k in RLAB and v in ("mask", "tag", "off")})
        self.policy = pol
        self.overlay = cfg.get("redact_display") == "overlay"
        self.h = None
        # stream state (in memory only)
        self.words = []                  # {"w", "t0", "t1", "conf", "spk", "final"}
        self.bounds = []                 # utterance ends (word index, exclusive), ascending
        self.u0 = 0                      # typed: first word of the open message
        self.t_u = 0.0
        self.sent = []                   # what the page shows per word (diff base)
        self.spans = []                  # session span store (C spans, page field names)
        self.masks = {}                  # i -> (cat, span start, word text): zero-lag redaction, never released while the word stays
        self.events = []
        self.utts = {}                   # end -> closed utterance read-outs
        self.ro = None
        self.lpts = []                   # lane points [j, lane, [[tag, p], ...]]
        self.lval = {}                   # lane -> last value
        self.ltags = {}                  # tag -> thr (sent)
        self.lanes_full = True
        self._newtags = {}
        self.turn_rev = collections.Counter()
        self.turn_sig = {}
        self.rev = 0
        self.seq = 0
        self.tag_ms = collections.deque(maxlen=200)
        self.final_t = collections.deque(maxlen=400)
        self.asr_lat = collections.deque(maxlen=200)
        self.red_mismatch = 0
        self.redlog = {}                 # run id -> {id, w0, w1, tag, type, conf, status, t0} (no text)
        self.redlog_sent = None
        self.q = asyncio.Queue()
        self.closed = False
        self.full = False
        self.last_in = time.monotonic()
        self.dg = None
        self.nf = 0                      # live: words before the interim tail
        self.audio_s = 0.0

    # ---------------------------------------------------------------- setup (worker thread)
    def open(self):
        self.h = self.app.pool.get(self.precision)
        vcfg = self.vert["cfg"]
        emit = list(vcfg.get("emit") or [])
        if self.red_on and emit:                                    # masked categories must be visible to mirror the redaction
            for c, v in self.policy.items():
                if v in ("mask", "tag") and CAT_FAMILY[c] not in emit:
                    emit.append(CAT_FAMILY[c])
        c = {k: v for k, v in vcfg.items() if k not in CONFIG_IGNORED and not k.startswith("_")}
        c.update({"emit": emit, "tags": vcfg.get("tags") or [], "thresholds": vcfg.get("thresholds") or {},
                  "domain": vcfg.get("domain", "auto"), "fine_auto": vcfg.get("fine_auto", True),
                  "asr": "deepgram" if self.provider == "deepgram" else vcfg.get("asr", "unknown"),
                  "stream_ctx": int(vcfg.get("stream_ctx", 256)), "max_ctx": int(vcfg.get("max_ctx", 2048))})
        mask = [k for k, v in self.policy.items() if v == "mask"]
        c["redact"] = {"mode": "mask", "policy": {k: ("mask" if k in mask else "off") for k in RCATS}} if (self.red_on and mask) else "off"
        self.h.apply_config(c)
        self.h.reset()
        self.info = self.h.info()

    def close_handle(self):
        if self.h is not None:
            h, self.h = self.h, None
            self.app.pool.put(h)

    # ---------------------------------------------------------------- input -> pushes
    def typed_words(self, text, send, spk):
        toks = text.split()[:TYPE_MAX_WORDS]
        done = send or (text[-1:].isspace() if text else False)
        ws = []
        for i, t in enumerate(toks):
            w = {"w": t, "t0": round(self.t_u + TYPE_WPS * i, 2), "t1": round(self.t_u + TYPE_WPS * i + 0.3, 2), "conf": 0.99,
                 "final": i < len(toks) - 1 or done}
            if spk is not None:
                w["spk"] = spk
            ws.append(w)
        return ws

    def replace_from(self, a, ws, endpoint):
        """words[a:] := ws (+ endpoint) -> (push object, first changed index)"""
        old = self.words[a:]
        k = 0
        while k < min(len(old), len(ws)) and old[k]["w"] == ws[k]["w"] and old[k]["final"] == ws[k]["final"] and old[k].get("spk") == ws[k].get("spk"):
            k += 1
        changed = a + k if (k < len(old) or k < len(ws)) else len(self.words)
        self.words[a:] = ws
        self.bounds = [b for b in self.bounds if b <= len(self.words)]
        if endpoint and ws and (not self.bounds or self.bounds[-1] < len(self.words)):
            self.bounds.append(len(self.words))
        cw = []
        for w in ws:
            x = {"w": w["w"], "t0": w["t0"], "t1": w["t1"], "conf": w.get("conf", 0.99), "final": w["final"]}
            if w.get("spk") is not None:
                x["spk"] = str(w["spk"])
            cw.append(x)
        return {"words": cw, "from": a, "endpoint": bool(endpoint)}, changed

    def on_typed(self, c):
        """returns (push, changed index, k, send) or None"""
        if self.closed:
            return None
        self.last_in = time.monotonic()
        text = str(c.get("text") or "")[:TYPE_MAX_CHARS]
        send = bool(c.get("send"))
        spk = c.get("spk") if isinstance(c.get("spk"), int) and 0 <= c.get("spk") < 100 else None
        ws = self.typed_words(text, send, spk)
        if self.u0 + len(ws) > SESSION_MAX_WORDS:
            self.full = True
            return None
        if send and not ws:
            return None
        push, changed = self.replace_from(self.u0, ws, send)
        if send:
            self.u0 = len(self.words); self.t_u = round(ws[-1]["t1"] + 1.0, 2)
        return push, changed

    def on_asr(self, ws, is_final, endpoint):
        """Deepgram Results: ws replace the interim tail; finals advance it"""
        if len(self.words) + len(ws) > SESSION_MAX_WORDS:
            self.full = True
            return None
        push, changed = self.replace_from(self.nf, ws, endpoint)
        if is_final:
            self.nf = len(self.words)
        return push, changed

    def on_endpoint(self):
        if self.words and (not self.bounds or self.bounds[-1] < len(self.words)):
            self.bounds.append(len(self.words))
            return {"endpoint": True}, len(self.words)
        return None

    # ---------------------------------------------------------------- one update (worker thread)
    def step(self, pushes, changed, type_k=None):
        t0 = time.perf_counter()
        for p in pushes:
            self.h.push(p)
        r = self.h.poll()
        ms = 1000 * (time.perf_counter() - t0)
        self.tag_ms.append(ms)
        return self.build(r, changed, type_k, ms)

    @staticmethod
    def conv_span(s):
        t = s["type"]
        fam = "toxicity" if t.startswith("safety.toxicity.") else s["family"]     # (the library reports these under "profanity")
        return {"tag": t, "fam": fam, "a": s["start"], "b": s["end"], "p": r3(s["score"]), "thr": r3(s.get("thr")), "src": s.get("src")}

    def build(self, r, changed, type_k, ms):
        n = len(self.words)
        self.seq += 1
        self.lpts = [p for p in self.lpts if p[0] < min(changed, n)]
        a0 = int(r.get("a0", 0))
        new = [self.conv_span(s) for s in r.get("spans", [])]
        # entity.numeral overlaps the number subtypes: keep it only where no subtype span covers the same words
        sub = [s for s in new if s["fam"] == "number" and s["tag"] != "entity.numeral"]
        new = [s for s in new if s["tag"] != "entity.numeral" or not any(x["a"] < s["b"] and x["b"] > s["a"] for x in sub)]
        self.spans = [s for s in self.spans if s["b"] <= a0 and s["b"] <= n] + new
        nfin = n if (n and self.words[-1]["final"]) else n - 1
        closed_to = self.bounds[-1] if self.bounds else 0
        t1 = [dict(s, tier="T1") for s in new if s["b"] < nfin or s["b"] <= closed_to]
        t0 = [dict(s, tier="T0") for s in new]

        # redaction: the C library's categories and policy, mirrored per word (checked against its redacted_text)
        disp = self.redact_display(n, new, a0, r.get("redacted_text"))
        cur = []
        for i, w in enumerate(self.words):
            d = disp[i]
            cur.append((d[0], bool(w["final"]), d[1], w.get("spk")))
        k = 0
        while k < min(len(self.sent), n) and self.sent[k] == cur[k]:
            k += 1
        words = []
        for i in range(k, n):
            w = self.words[i]; d = disp[i]
            x = {"w": w["w"] if self.overlay else d[0], "f": bool(w["final"]), "t0": w.get("t0"), "t1": w.get("t1"), "spk": w.get("spk")}
            if self.overlay:
                x["rx"] = d[0]
            if d[1]:
                x["r"] = d[1]; x["rid"] = d[2]
            words.append(x)
        nf_new = sum(1 for x in cur if x[1]) - sum(1 for x in self.sent if x[1])
        now = time.time()
        for _ in range(max(0, nf_new)):
            self.final_t.append(now)
        self.sent = cur

        # events (T0 display tier of the library), deduplicated
        self.events = [e for e in self.events if e["i"] < n]
        ev_new = []
        for e in r.get("events", []):
            x = {"tag": e["type"], "q": e.get("q"), "i": e["word"], "p": r3(e["score"]), "thr": r3(e.get("thr")), "tier": "T0",
                 "label": self.app.label(e["type"])}
            if x["i"] < n and not any(y["tag"] == x["tag"] and y["i"] == x["i"] for y in self.events):
                self.events.append(x); ev_new.append(x)

        # closed utterances
        utt_done, alerts = [], []
        for u in r.get("utterances", []):
            ro = {k2: v for k2, v in u.items() if k2 not in ("text", "start", "end", "final")}
            self.utts[u["end"]] = {"w0": u["start"], "w1": u["end"], "readouts": ro}
            utt_done.append({"w0": u["start"], "w1": u["end"], "readouts": ro})
            for s in ro.get("safety", []):
                if s.get("on"):
                    alerts.append({"kind": "utterance", "tag": s["tag"], "p": r3(s["p"]), "thr": r3(s["thr"]), "w0": u["start"], "w1": u["end"]})
            self.lane_points(u["end"] - 1, ro, u["start"])
        for e in list(self.utts):
            if e > n or (self.bounds and e not in self.bounds):
                del self.utts[e]
        cur_ro = r.get("current")
        if cur_ro:
            self.ro = dict({k2: v for k2, v in cur_ro.items() if k2 not in ("text", "start", "end")}, j=n - 1, final=False)
            self.lane_points(n - 1, cur_ro, cur_ro["start"])
        elif self.bounds and self.bounds[-1] in self.utts:
            self.ro = dict(self.utts[self.bounds[-1]]["readouts"], j=self.bounds[-1] - 1, final=True)
        lanes = self.lanes_msg(changed, n)

        typed = None
        if self.provider == "typed":
            src = cur_ro if cur_ro else (self.utts.get(self.bounds[-1], {}).get("readouts") if self.bounds else None)
            on = []
            if src:
                on = [dict(x, kind="safety") for x in src.get("safety", []) if x.get("on")]
                on += [dict(x, kind="intent") for x in src.get("intent", []) if x["tag"] == "intent.warn_threaten" and x.get("on")]
            typed = {"j": (n - 1) if n else None, "on": on}

        recent = [t for t in self.final_t if now - t <= 15]
        stats = {"n_words": n, "profile": r.get("profile"), "precision": self.precision, "tag_ms": round(ms, 1),
                 "word_p50": r3(pctl(list(self.tag_ms), 50)), "word_p95": r3(pctl(list(self.tag_ms), 95)),
                 "words_per_s": round(len(recent) / 15.0, 2) if recent else 0.0, "redact_mismatch": self.red_mismatch,
                 "asr_lat_p50": r3(pctl(list(self.asr_lat), 50)), "audio_s": round(self.audio_s, 1)}
        self.rev += 1
        upd = {"type": "update", "seq": self.seq, "from": k, "n": n, "words": words,
               "tiers": {"T0": {"a0": a0, "spans": t0}, "T1": {"a0": a0, "spans": t1}},
               "events": ev_new, "alerts": alerts, "readouts": self.ro or {}, "utt_done": utt_done, "utt_bounds": self.bounds[-50:],
               "stats": stats, "redact": self.redact_info(disp),
               "enrich": self.enrich(n, changed), "lanes": lanes, "server_ms": round(ms, 1)}
        if typed is not None:
            upd["typed"] = typed
        if type_k is not None:
            upd["type_k"] = type_k
        return upd

    # ---------------------------------------------------------------- redaction
    def redact_display(self, n, new, a0, c_text):
        """per word (text shown, None | mask | tag, run id). Same span -> category rule as the library's redact_text (first span in
        (start, -end, type) order wins a word; one placeholder per run of words of the same category and span start)"""
        out = [(w["w"], None, None) for w in self.words]
        if not self.red_on or not n:
            return out
        cat = [None] * n
        for s in sorted(self.spans, key=lambda s: (s["a"], -s["b"], s["tag"])):
            c = TAG_CAT.get(s["tag"])
            if c is None or self.policy.get(c, "off") == "off":
                continue
            for i in range(max(0, s["a"]), min(s["b"], n)):
                if cat[i] is None:
                    cat[i] = (c, s["a"])
        if c_text is not None and self.red_mismatch < 1000:     # cross-check with the library over its window [a0, n)
            if self.c_redacted(cat, a0, n, mask_only=True) != c_text:
                self.red_mismatch += 1
        for i in list(self.masks):                              # zero-lag hold: a masked word stays masked while it is unchanged
            c, st, wt = self.masks[i]
            if i >= n or self.words[i]["w"] != wt:
                del self.masks[i]
            elif cat[i] is None:
                cat[i] = (c, st)
        for i in range(n):
            if cat[i] is not None and self.policy.get(cat[i][0]) == "mask":
                self.masks[i] = (cat[i][0], cat[i][1], self.words[i]["w"])
        i = 0
        while i < n:
            if cat[i] is None:
                i += 1; continue
            j = i
            while j < n and cat[j] == cat[i]:
                j += 1
            c, st = cat[i]
            rid = f"{c}:{st}"
            for t in range(i, j):
                if self.policy.get(c) == "mask":
                    out[t] = (("[" + RLAB[c] + "]") if t == i else "", "mask", rid)
                else:
                    out[t] = (self.words[t]["w"], "tag", rid)
            i = j
        return out

    def redact_info(self, disp):
        """redaction counters + the masked-span log of the page's redaction panel (categories, word ranges, scores; no text)"""
        n = len(disp)
        runs = {}
        for i, d in enumerate(disp):
            if d[1] == "mask":
                r = runs.setdefault(d[2], [i, i + 1]); r[1] = i + 1
        for rid in list(self.redlog):
            if rid not in runs:
                del self.redlog[rid]
        for rid, (a, b) in runs.items():
            cat = rid.split(":")[0]
            sp = max((s for s in self.spans if TAG_CAT.get(s["tag"]) == cat and s["a"] < b and s["b"] > a), key=lambda s: s["p"] or 0, default=None)
            e = self.redlog.get(rid) or {"id": len(self.redlog) + 1, "t0": self.words[a].get("t0") if a < n else None, "status": "masked"}
            e.update({"w0": a, "w1": b, "type": RLAB[cat], "cat": cat, "tag": sp["tag"] if sp else None, "conf": sp["p"] if sp else None})
            self.redlog[rid] = e
        out = {"counters": {"spans": len(runs), "masked_words": sum(b - a for a, b in runs.values())},
               "display": "overlay" if (self.red_on and self.overlay) else ("redacted" if self.red_on else "off")}
        log_ = sorted(self.redlog.values(), key=lambda e: e["w0"])
        sig = json.dumps(log_)
        if sig != self.redlog_sent:
            self.redlog_sent = sig
            out["log"] = log_
        return out

    def export(self):
        """the redacted transcript of the session (the page's "Export redacted JSON")"""
        disp = list(self.sent)                                   # what the page shows: (text, final, redaction state, spk)
        return {"words": [{"w": d[0], "t0": w.get("t0"), "t1": w.get("t1"), "spk": w.get("spk"), **({"r": d[2]} if d[2] else {})}
                          for d, w in zip(disp, self.words) if d[0]],
                "redaction": {"on": self.red_on, "policy": self.policy}, "log": sorted(self.redlog.values(), key=lambda e: e["w0"]),
                "utterances": [dict(w0=u["w0"], w1=u["w1"]) for u in sorted(self.utts.values(), key=lambda u: u["w0"])]}

    def c_redacted(self, cat, a, b, mask_only):
        parts, i = [], a
        while i < b:
            if cat[i] is None or (mask_only and self.policy.get(cat[i][0]) != "mask"):
                parts.append(self.words[i]["w"]); i += 1; continue
            j = i
            while j < b and cat[j] == cat[i]:
                j += 1
            parts.append("[" + RLAB[cat[i][0]] + "]"); i = j
        return " ".join(parts)

    # ---------------------------------------------------------------- lanes
    @staticmethod
    def lane_values(ro):
        lanes = {"intent": ro.get("intent") or [], "act": ro.get("act") or [], "safety": ro.get("safety") or []}
        for g, lst in (ro.get("state") or {}).items():
            lanes[g] = lst
        out = {}
        for lane, lst in lanes.items():
            v = sorted(((x["tag"], float(x["p"]), float(x["thr"])) for x in lst if x["p"] >= 0.5 * x["thr"]), key=lambda t: -t[1])[:3]
            out[lane] = v
        return out

    def lane_points(self, j, ro, ustart):
        if j < 0:
            return
        self.lpts = [p for p in self.lpts if p[0] < j]
        prev = {}
        for p in self.lpts:
            prev[p[1]] = p[2]
        for lane, v in self.lane_values(ro).items():
            val = [[t, snap(p)] for t, p, _ in v]
            for t, _, thr in v:
                self.ltags.setdefault(t, None)
                if self.ltags[t] is None or self.ltags[t] != r3(thr):
                    self.ltags[t] = r3(thr); self._newtags[t] = r3(thr)
            old = prev.get(lane)
            same = old is not None and [x[0] for x in old] == [x[0] for x in val] and all(abs(a[1] - b[1]) < 0.1 for a, b in zip(old, val))
            if j == ustart or not same:
                self.lpts.append([j, lane, val])

    def lanes_msg(self, changed, n):
        lo = min(changed, n)
        self.lpts = [p for p in self.lpts if p[0] < n]
        full = self.lanes_full
        self.lanes_full = False
        pts = [p for p in self.lpts if full or p[0] >= lo]
        tags = dict(self.ltags) if full else dict(self._newtags)
        self._newtags = {}
        return {"from": 0 if full else lo, "n": n, "full": full, "pts": pts, "tags": tags}

    # ---------------------------------------------------------------- turns (one per utterance) for the page's enrich view
    def dist(self, lst, prefix_ok):
        L = [x for x in lst if prefix_ok(x["tag"])]
        if not L:
            return None
        top = L[0]
        return {"top": top["tag"], "label": self.app.label(top["tag"]), "p": r3(top["p"]), "p_raw": r3(top["p"]), "mass": None,
                "thr": r3(top["thr"]), "on": bool(top["on"]), "dist": [{"tag": x["tag"], "label": self.app.label(x["tag"]), "p": r3(x["p"])} for x in L]}

    def turn(self, seq, a, b, open_):
        ro = (self.ro if open_ else (self.utts.get(b) or {}).get("readouts")) or {}
        act = self.dist(ro.get("act") or [], lambda t: t.startswith("discourse.act."))
        intents = ro.get("intent") or []
        it = intents[0] if intents else None
        role = self.dist(ro.get("role") or [], lambda t: True) if any(self.words[i].get("spk") is not None for i in range(a, b)) else None
        spans = [s for s in self.spans if a <= s["a"] < b]
        t = {"id": f"t{seq}", "seq": seq, "a": a, "b": b, "open": open_,
             "final": (not open_) and all(self.words[i]["final"] for i in range(a, b)),
             "src": "start" if seq == 0 else "asr_utt_end", "spk": self.words[a].get("spk") if a < len(self.words) else None, "slot": 0,
             "readout": "interim" if open_ else "turn_end", "role": role,
             "role_note": None if role else "speaker role needs speaker ids on the words",
             "act": act, "intent": ({"tag": it["tag"], "label": self.app.label(it["tag"]), "p": r3(it["p"]), "thr": r3(it["thr"]), "on": bool(it["on"])} if it else None),
             "spans": [{"tag": s["tag"], "label": self.app.label(s["tag"]), "fam": "entity", "a": s["a"], "b": s["b"], "p": s["p"], "tier": "T1"}
                       for s in spans if s["fam"] == "entity"],
             "numbers": [{"tag": s["tag"], "label": self.app.label(s["tag"]), "fam": "number", "a": s["a"], "b": s["b"], "p": s["p"], "tier": "T1"}
                         for s in spans if s["fam"] == "number"],
             "safety": [{"tag": s["tag"], "label": self.app.label(s["tag"]), "kind": "span", "a": s["a"], "b": s["b"], "p": s["p"], "tier": "T1"}
                        for s in spans if s["fam"] in ("pii", "profanity", "toxicity")] +
                       [{"tag": x["tag"], "label": self.app.label(x["tag"]), "kind": "utt", "p": r3(x["p"]), "thr": r3(x["thr"]), "on": True}
                        for x in ro.get("safety") or [] if x.get("on")],
             "events": [{"tag": e["tag"], "label": e["label"], "i": e["i"], "p": e["p"], "thr": e["thr"], "tier": e["tier"]}
                        for e in self.events if a <= e["i"] < b]}
        if open_ and b > a and t["spans"] is not None:
            for s in t["spans"] + t["numbers"]:
                s["tier"] = "T1" if s["b"] < b else "T0"
        return t

    def enrich(self, n, changed):
        segs, a = [], 0
        for b in self.bounds:
            if b > a:
                segs.append((a, b, False)); a = b
        if a < n:
            segs.append((a, n, True))
        up = []
        for seq, (a, b, op) in enumerate(segs):
            t = self.turn(seq, a, b, op)
            sig = json.dumps(t, sort_keys=True)
            if self.turn_sig.get(seq) != sig:
                self.turn_sig[seq] = sig; self.turn_rev[seq] += 1
                t["rev"] = self.turn_rev[seq]
                up.append(t)
        for s in list(self.turn_sig):
            if s >= len(segs):
                del self.turn_sig[s]
        v = self.vert
        return {"schema": 2, "rev": self.rev, "vertical": v["id"],
                "domain": {"active": v["id"], "requested": v["id"], "detected": None, "conf": 1.0, "locked": True, "candidate": None,
                           "codes": [v["domain"]], "profile": self.info.get("profile") if getattr(self, "info", None) else None},
                "turns": {"n": len(segs), "upsert": up}, "speakers": [],
                "live": {"i": n - 1, "act": self.dist((self.ro or {}).get("act") or [], lambda t: t.startswith("discourse.act.")),
                         "intent": None, "flags": {}, "items": {}},
                "state": {}, "state_changes": [], "sections": []}


# ------------------------------------------------------------------------------------------------------------ application
class App:
    def __init__(self, args):
        self.args = args
        self.model_dir = Path(args.model).resolve()
        self.lib = AF.Lib(AF.find_lib(ROOT, args.lib))
        self.verts = Verticals(Path(args.configs))
        self.labels = json.loads((HERE / "labels.json").read_text())
        self.int8_ok = (self.model_dir / "arctag_int8.onnx").is_file()
        self.precision = "int8" if args.int8 else "fp32"
        if self.precision == "int8" and not self.int8_ok:
            sys.exit(f"--int8: {self.model_dir}/arctag_int8.onnx not found (ARCTAG_FETCH=int8 scripts/fetch_model.sh)")
        self.pool = Pool(self.lib, self.model_dir, args.threads)
        self.exe = concurrent.futures.ThreadPoolExecutor(max_workers=max(2, args.max_sessions))
        self.sessions = set()
        self.dg_key = os.environ.get("DEEPGRAM_API_KEY", "").strip() or None
        h = self.pool.get(self.precision)                         # fail fast (library, model files) and warm one handle
        self.version = h.info().get("version")
        self.pool.put(h)

    def label(self, tag):
        if tag in self.labels:
            return self.labels[tag]
        t = tag
        for pre in ("safety.pii.", "safety.profanity.", "safety.toxicity.", "safety.", "entity.mention.", "entity.number.", "discourse.intent.",
                    "discourse.act.", "discourse.", "intent.", "event.sports.", "event.legal.", "event.meeting.", "content.", "speaker.role.", "affect."):
            if t.startswith(pre):
                t = t[len(pre):]; break
        t = t.replace(".recent", "").replace("_", " ").replace(".", " ")
        return t[:1].upper() + t[1:]

    def hello(self):
        return {"type": "hello", "schema": 2, "version": self.version, "backend": "arctag C library (ctypes)",
                "precision": self.precision, "precisions": ["fp32"] + (["int8"] if self.int8_ok else []),
                "verticals": [{k: v for k, v in x.items() if k in ("id", "label", "config", "domain")} for x in self.verts.public()],
                "profiles": self.verts.public(), "labels": self.labels,
                "type_mode": {"provider": "typed", "guard": {"enabled": False, "model": None, "loaded": False, "error": None, "thr": None}},
                "live": {"enabled": bool(self.dg_key), "provider": "Deepgram nova-3" if self.dg_key else None,
                         "note": None if self.dg_key else "Live mic needs a Deepgram key: start the server with DEEPGRAM_API_KEY set."},
                "limits": {"type_max_words": TYPE_MAX_WORDS, "session_max_words": SESSION_MAX_WORDS, "max_sessions": self.args.max_sessions},
                "server": {"sessions": len(self.sessions)}}

    # ---------------------------------------------------------------- HTTP (static files + small JSON endpoints)
    def http(self, connection, request):
        path = request.path.split("?", 1)[0]
        if path.endswith("/ws") or path == "/ws":
            return None                                            # WebSocket handshake
        if path in ("/api/health", "/health"):
            return self.json_response({"ok": True, "version": self.version, "sessions": len(self.sessions)})
        if path == "/api/profiles":
            return self.json_response(self.verts.public())
        rel = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        if rel.startswith("static/"):
            rel = rel[len("static/"):]
        f = (STATIC / rel).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            return self.text_response(404, "not found")
        ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        return Response(200, "OK", Headers([("Content-Type", ctype), ("Cache-Control", "no-cache"), ("Content-Length", str(f.stat().st_size)),
                                             ("X-Content-Type-Options", "nosniff")]), f.read_bytes())

    @staticmethod
    def json_response(obj):
        b = json.dumps(obj).encode()
        return Response(200, "OK", Headers([("Content-Type", "application/json"), ("Cache-Control", "no-store"), ("Content-Length", str(len(b)))]), b)

    @staticmethod
    def text_response(code, txt):
        b = txt.encode()
        return Response(code, "Not Found" if code == 404 else "Error", Headers([("Content-Type", "text/plain"), ("Content-Length", str(len(b)))]), b)

    # ---------------------------------------------------------------- WebSocket
    async def ws(self, conn):
        st = {"sess": None}

        async def send(o):
            try:
                await conn.send(json.dumps(o, separators=(",", ":")))
            except websockets.ConnectionClosed:
                pass

        await send(self.hello())
        try:
            async for raw in conn:
                S = st["sess"]
                if isinstance(raw, bytes):
                    if S is not None and S.dg is not None:
                        S.audio_s += len(raw) / 32000.0
                        await S.dg.audio(raw)
                    continue
                try:
                    c = json.loads(raw)
                    if not isinstance(c, dict):
                        continue
                except ValueError:
                    continue
                t = c.get("type")
                try:
                    if t == "start":
                        await self.stop(st)
                        st["sess"] = await self.start(conn, send, c.get("config") or {})
                    elif t == "type_text" and S is not None and S.provider == "typed" and not S.closed:
                        res = S.on_typed(c)
                        if S.full and not getattr(S, "full_told", False):
                            S.full_told = True
                            await send({"type": "status", "level": "error", "msg": f"session is full ({SESSION_MAX_WORDS} words): start a new conversation"})
                        if res is not None:
                            await S.q.put(("push", res[0], res[1], c.get("k") if isinstance(c.get("k"), int) else None))
                        if c.get("send") and c.get("guard", True) is not False:
                            await send({"type": "guard", "k": c.get("k"), "model": None, "verdict": None, "disabled": True})
                    elif t == "finish" and S is not None:
                        await self.stop(st, "finished")             # live: flushes Deepgram and tags its last words first
                    elif t == "stop":
                        await self.stop(st, "user_stop")
                        await send({"type": "stopped"})
                    elif t == "redact" and S is not None:
                        await S.q.put(("redact", c))
                    elif t == "vertical" and S is not None:
                        await send({"type": "status", "level": "info", "msg": "the vertical applies from the next session (press Stop, then start again)"})
                    elif t == "status":
                        await send(self.hello())
                    elif t == "export" and S is not None:
                        await send({"type": "export", "data": S.export()})
                except Exception as e:                            # one bad message never kills the connection
                    log(f"[ws] {t}: {type(e).__name__}")
                    await send({"type": "status", "level": "warn", "msg": f"server could not handle '{t}': {type(e).__name__}"})
        except websockets.ConnectionClosed:
            pass
        finally:
            await self.stop(st, "ws_closed", notify=False)

    async def start(self, conn, send, cfg):
        if len(self.sessions) >= self.args.max_sessions:
            await send({"type": "error", "code": "limit", "msg": f"{self.args.max_sessions} sessions are already running: try again in a minute"})
            return None
        S = Session(self, conn, cfg)
        S.send = send
        loop = asyncio.get_running_loop()
        if S.provider == "deepgram":
            if not self.dg_key:
                await send({"type": "error", "code": "auth", "msg": "live ASR is off: start the server with DEEPGRAM_API_KEY set"})
                return None
        try:
            await loop.run_in_executor(self.exe, S.open)
        except Exception as e:
            log(f"[session] open failed: {type(e).__name__}")
            await send({"type": "error", "msg": f"session init failed: {e}"[:300]})
            return None
        if S.provider == "deepgram":
            import deepgram_live as DG
            S.dg = DG.DeepgramLive(self.dg_key, cfg, lambda ws, fin, ep, lat: S.q.put_nowait(("asr", ws, fin, ep, lat)),
                                   lambda: S.q.put_nowait(("endpoint",)), lambda st_: asyncio.ensure_future(send(dict({"type": "status"}, **st_))))
            try:
                await S.dg.start()
            except Exception as e:
                log(f"[session] deepgram connect failed: {type(e).__name__}")
                await loop.run_in_executor(self.exe, S.close_handle)
                await send({"type": "error", "msg": f"Deepgram: could not connect ({type(e).__name__})"})
                return None
        self.sessions.add(S)
        S.worker = asyncio.create_task(self.consume(S))
        if S.provider == "typed":
            S.idle = asyncio.create_task(self.idle(S))
        await send({"type": "started", "provider": S.provider, "needs_audio": S.provider != "typed", "sid": S.sid, "profile": S.vert["id"],
                    "schema": 2, "precision": S.precision, "config": S.vert["config"],
                    "info": {k: v for k, v in S.info.items() if k != "model_file"}})
        log(f"[session] start {S.sid} provider={S.provider} vertical={S.vert['id']} precision={S.precision} (sessions={len(self.sessions)})")
        return S

    async def consume(self, S):
        """one worker per session: drains everything queued, pushes it all, polls once (coalescing while the model runs)"""
        loop = asyncio.get_running_loop()
        try:
            while not S.closed:
                items = [await S.q.get()]
                while not S.q.empty():
                    items.append(S.q.get_nowait())
                stop = any(x[0] == "end" for x in items)
                if stop:
                    items = items[:[x[0] for x in items].index("end")]
                pushes, changed, k = [], 10 ** 9, None
                for it in items:
                    if it[0] == "push":
                        pushes.append(it[1]); changed = min(changed, it[2]); k = it[3] if it[3] is not None else k
                    elif it[0] == "asr":
                        res = S.on_asr(it[1], it[2], it[3])
                        if it[4] is not None:
                            S.asr_lat.append(it[4])
                        if res:
                            pushes.append(res[0]); changed = min(changed, res[1])
                    elif it[0] == "endpoint":
                        res = S.on_endpoint()
                        if res:
                            pushes.append(res[0]); changed = min(changed, res[1])
                    elif it[0] == "redact":
                        c = it[1]
                        if "mode" in c:
                            S.red_on = c["mode"] not in ("off", None, "")
                        if c.get("display"):
                            S.overlay = c["display"] == "overlay"
                        if isinstance(c.get("policy"), dict) and c["policy"]:
                            S.policy = dict(DEFAULT_POLICY); S.policy.update({a: b for a, b in c["policy"].items() if a in RLAB})
                        S.masks = {}; S.sent = []
                        await loop.run_in_executor(self.exe, S.open_redact)
                        changed = min(changed, 0); pushes.append(None)
                pushes = [p for p in pushes if p is not None]
                if changed == 10 ** 9 and not pushes:
                    if stop:
                        break
                    continue
                if changed == 10 ** 9:
                    changed = len(S.words)
                upd = await loop.run_in_executor(self.exe, S.step, pushes, changed, k)
                await S.send(upd)
                if stop:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log(f"[session] {S.sid} tagger error: {type(e).__name__}")
            await S.send({"type": "status", "level": "error", "msg": f"tagger error: {type(e).__name__}"})

    async def idle(self, S):
        try:
            while not S.closed:
                await asyncio.sleep(10)
                if time.monotonic() - S.last_in > TYPE_IDLE_S:
                    S.closed = True
                    await S.send({"type": "done", "reason": "type_idle"})
                    await self.end(S)
                    return
        except asyncio.CancelledError:
            pass

    async def end(self, S):
        if S in self.sessions:
            self.sessions.discard(S)
        S.closed = True
        if S.dg is not None:
            await S.dg.close()
        w = getattr(S, "worker", None)
        if w is not None and w is not asyncio.current_task():
            S.q.put_nowait(("end",))
            try:
                await asyncio.wait_for(w, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                w.cancel()
        i = getattr(S, "idle", None)
        if i is not None and i is not asyncio.current_task():
            i.cancel()
        await asyncio.get_running_loop().run_in_executor(self.exe, S.close_handle)
        S.words = []; S.sent = []; S.spans = []
        log(f"[session] end {S.sid} (sessions={len(self.sessions)})")

    async def stop(self, st, reason="stop", notify=True):
        S = st.get("sess")
        st["sess"] = None
        if S is None:
            return
        if S.dg is not None and reason == "finished":
            await S.dg.finish()
        await self.end(S)
        if notify and reason not in ("user_stop",):
            await S.send({"type": "done", "reason": reason})


def _open_redact(self):
    """re-apply the redaction option after a mid-session change (stream kept)"""
    mask = [k for k, v in self.policy.items() if v == "mask"]
    if self.red_on and mask:
        self.h.set_option("redact", ",".join(f"{k}=mask" for k in mask))
    else:
        self.h.set_option("redact", "off")


Session.open_redact = _open_redact


def free_port():
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


async def amain(args):
    logging.getLogger("websockets").setLevel(logging.CRITICAL)    # (browsers' speculative connections end as failed handshakes)
    app = App(args)
    port = args.port or free_port()
    async with serve(app.ws, args.host, port, process_request=app.http, max_size=2 ** 22, ping_interval=20, ping_timeout=30,
                     server_header=None) as srv:
        log(f"Arc Tagger demo on http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{port}/  "
            f"(arctag {app.version}, {app.precision}, model {app.model_dir.name}/, live ASR {'on' if app.dg_key else 'off'})")
        if args.port_file:
            Path(args.port_file).write_text(str(port))
        await srv.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Arc Tagger web demo (arctag C library)")
    ap.add_argument("--host", default=os.environ.get("ARCTAG_DEMO_HOST", "127.0.0.1"), help="bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("ARCTAG_DEMO_PORT", "8080")), help="port (default 8080; 0 = a free port)")
    ap.add_argument("--model", default=os.environ.get("ARCTAG_MODEL_DIR", str(ROOT / "model")), help="model directory (default ./model)")
    ap.add_argument("--configs", default=str(ROOT / "configs"), help="vertical configs directory (default ./configs)")
    ap.add_argument("--lib", default=None, help="path of libarctag.so / .dylib / arctag.dll (default: build/)")
    ap.add_argument("--int8", action="store_true", help="default precision int8 (the page can still switch per session)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("ARCTAG_THREADS", "2")), help="ONNX Runtime threads per session (default 2)")
    ap.add_argument("--max-sessions", type=int, default=8, help="concurrent sessions (each holds one model session, ~110 MB fp32)")
    ap.add_argument("--port-file", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
