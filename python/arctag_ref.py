"""Python reference for the arctag C library (numpy + onnxruntime + tokenizers).

ONNX inference only: the same input features, ONNX graph and decoding as the C library (the T0, zero-lag tier):
  text -> whitespace words -> utterances (a word ending in . ? ! closes one) -> per-word input features -> arctag.onnx
  (or arctag_int8.onnx) -> per-word / per-utterance probabilities -> spans (per-family span output + span_in runs), events (ev_at / ev_imm / ev_rec, 3-word de-dup,
  display tier T0, hidden events dropped), utterance read-outs (intent / act / safety / emotion / arousal / valence /
  intensity / item / state / role) with the per-profile display thresholds of arctag_meta.json. Optional: threshold
  overrides, redaction, output filters (the same options as the C library).

  python arctag_ref.py MODEL_DIR "some text" [domain]   -> JSON
  python arctag_ref.py MODEL_DIR --tokens "some text"   -> token ids per word
  ARCTAG_PRECISION=int8 selects arctag_int8.onnx
"""
import json, math, os, re, sys
import numpy as np

WS = re.compile(r"[ \t\n\r\f\v]+")
SENT_END = (".", "?", "!")
P_MAX = 8
FAMILY = [("pii", ("safety.pii.",)), ("profanity", ("safety.profanity.", "safety.toxicity.")),
          ("number", ("entity.number.", "entity.numeral")), ("entity", ("entity.mention.",))]
DECODE_FAM = [("number", ("entity.number.",)), ("entity", ("entity.mention.",)), ("pii", ("safety.pii.",)),
              ("profanity", ("safety.profanity.",)), ("toxicity", ("safety.toxicity.",))]
UMBRELLA = ("entity.numeral", "safety.toxicity.toxic_any")
# redaction categories (span tag -> category -> placeholder); default policy = mask everything except public contacts
REDACT_CAT = {
    "safety.pii.phone_number": "phone", "safety.pii.email": "email", "safety.pii.street_address": "address",
    "safety.pii.card_account_number": "card", "safety.pii.gov_id": "gov_id", "safety.pii.date_of_birth": "dob",
    "safety.pii.private_name": "name", "safety.pii.url_handle": "handle", "safety.pii.public_contact": "contact",
    "safety.profanity.mild": "profanity", "safety.profanity.strong": "profanity",
    "safety.toxicity.insult_harassment": "toxicity", "safety.toxicity.hate_identity": "toxicity",
    "safety.toxicity.threat_violence": "toxicity", "safety.toxicity.sexual_explicit": "toxicity", "safety.toxicity.self_harm": "toxicity",
}
REDACT_LABEL = {"phone": "PHONE", "email": "EMAIL", "address": "ADDRESS", "card": "CARD", "gov_id": "GOV_ID", "dob": "DOB",
                "name": "NAME", "handle": "HANDLE", "contact": "CONTACT", "profanity": "PROFANITY", "toxicity": "TOXIC"}
REDACT_DEFAULT = {c: ("off" if c == "contact" else "mask") for c in REDACT_LABEL}
READOUT_KEYS = ("intent", "act", "safety", "emotion", "arousal", "valence", "intensity", "role", "item", "state")
SPAN_SECTIONS = ("entity", "number", "pii", "profanity", "toxicity")


def span_section(tag):
    return "toxicity" if tag.startswith("safety.toxicity.") else family_of(tag)


def thr_match(pat, tag):
    """override pattern: exact tag, or a prefix ending in '*'"""
    return tag.startswith(pat[:-1]) if pat.endswith("*") else tag == pat


def family_of(tag):
    for fam, pre in FAMILY:
        if tag.startswith(pre):
            return fam
    return "other"


def decode_family(tag):
    if tag in UMBRELLA:
        return None
    for fam, pre in DECODE_FAM:
        if tag.startswith(pre):
            return fam
    return None


def split_words(text):
    return [w for w in WS.split(text) if w]


def utt_starts(words):
    us, nxt = [], True
    for w in words:
        us.append(1 if nxt else 0)
        nxt = w.rstrip("\"')]}").endswith(SENT_END)
    return us


def _bk(x, edges):
    return 1 + sum(1 for e in edges if e <= x)


def _r6(x):
    return round(float(x), 6)


class ArcTagger:
    def __init__(self, model_dir, threads=1, precision=None, model_file=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.meta = json.load(open(os.path.join(model_dir, "arctag_meta.json")))
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        precision = precision or os.environ.get("ARCTAG_PRECISION", "fp32")
        if precision not in ("fp32", "int8"):
            raise ValueError("precision must be fp32 or int8")
        self.precision = precision
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        fn = model_file or ("arctag_int8.onnx" if precision == "int8" else "arctag.onnx")
        self.sess = ort.InferenceSession(os.path.join(model_dir, fn), so, providers=["CPUExecutionProvider"])
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.cn = self.meta["readouts"]
        self.asr_sources = self.meta["asr_sources"]
        ev = self.meta.get("events") or {}
        self.ev_skip = set(ev.get("hide") or []) | {t for t, v in (ev.get("tier") or {}).items() if v != "T0"}
        self.overrides = []          # [(pattern, readout or None, value)] in order; "head:<family>" for span-head families
        self.redact = None           # None = off, else {category: "mask" | "tag" | "off"}
        self.emit = None             # None = everything, else a set of sections
        self.tags = None             # None = every tag, else a list of tag prefixes
        self.fine_auto = True

    # ---------------------------------------------------------------- options (= the C library's)
    def set_threshold(self, pattern, value):
        q = None
        if "@" in pattern and not pattern.startswith("head:"):
            pattern, q = pattern.rsplit("@", 1)
        self.overrides.append((pattern, q, float(value)))

    def apply_config(self, cfg):
        """the runtime keys of a configs/*.json file (as arctag_apply_config; precision / threads are load-time here)"""
        for k, v in cfg.items():
            if k == "thresholds":
                self.overrides = []
                for pat, x in v.items():
                    self.set_threshold(pat, x)
            elif k == "redact":
                if isinstance(v, str):
                    v = {"mode": "off"} if v == "off" else {"mode": "mask"} if v in ("mask", "on") else \
                        {"mode": "mask", "policy": dict(x.split("=") for x in v.replace(" ", ",").split(",") if x)}
                if v.get("mode", "off") == "off":
                    self.redact = None
                else:
                    pol = v.get("policy")
                    self.redact = {c: ("mask" if pol.get(c) == "mask" else "off") for c in REDACT_LABEL} if pol else dict(REDACT_DEFAULT)
            elif k == "emit":
                self.emit = None if not v or set(v) >= set(SPAN_SECTIONS) | {"events"} | set(READOUT_KEYS) else set(v)
            elif k == "tags":
                self.tags = list(v) or None
            elif k == "fine_auto":
                self.fine_auto = bool(v)

    def _thresholds(self, prof):
        """the profile's thresholds with the overrides applied (a copy)"""
        if not self.overrides:
            return prof
        th = {q: list(v) for q, v in prof["thr"].items()}; ht = dict(prof["head_thr"])
        for pat, q0, v in self.overrides:
            if pat.startswith("head:"):
                if pat[5:] in ht:
                    ht[pat[5:]] = v
                continue
            for q, names in self.cn.items():
                if q0 is None or q0 == q:
                    for c, t in enumerate(names):
                        if thr_match(pat, t):
                            th[q][c] = v
            if q0 is None:
                for fam in ht:
                    if pat == fam or (pat.endswith("*") and (fam + ".").startswith(pat[:-1])):
                        ht[fam] = v
        return dict(prof, thr=th, head_thr=ht)

    # ---------------------------------------------------------------- inputs
    def pieces(self, w):
        ids = self.tok.encode(" " + w.strip(), add_special_tokens=False).ids or [0]
        return ids[:P_MAX - 1] + ids[-1:] if len(ids) > P_MAX else ids

    def profile(self, domain):
        m = self.meta
        key = domain if domain in m["profiles"] else m["aliases"].get(domain or "auto", m["default_profile"])
        return key, m["profiles"][key]

    def fine_hint(self, ui, words):
        """the fine domain the words' lexicon cues point to within a UI domain with cues (sports / court), or None"""
        cues = (self.meta.get("fine_cues") or {}).get(ui)
        if not cues or not words:
            return None
        ix = {}
        for fine, cs in cues.items():
            for c in cs:
                ix[tuple(c.replace("-", " ").split())] = fine
        toks = ["".join(ch for ch in w["w"].lower() if ch.isascii() and ch.isalnum()) for w in words[-3000:]]
        cnt = {}
        for i in range(len(toks)):
            for L in (1, 2, 3):
                f = ix.get(tuple(toks[i:i + L])) if i + L <= len(toks) else None
                if f:
                    cnt[f] = cnt.get(f, 0) + 1
        if not cnt:
            return None
        r = sorted(cnt.items(), key=lambda x: -x[1])
        if r[0][1] >= self.meta["fine_min"] and (len(r) == 1 or r[0][1] >= self.meta["fine_ratio"] * r[1][1]):
            return r[0][0]
        return None

    def features(self, words, asr):
        """words: dicts {w, t0?, t1?, conf?, spk?, final?} -> per-word (pieces, wfl, feats, spk slot), the graph inputs"""
        asr = asr or self.meta.get("default_asr", "unknown")
        a = asr if asr in self.asr_sources else "unknown"
        asr_i = self.asr_sources.index(a)
        spksrc = 2 if any(w.get("spk") is not None for w in words) else 0
        prev, slots, out = None, {}, []
        m = self.meta
        for w in words:
            t0, t1 = w.get("t0"), w.get("t1")
            last_t1 = prev["t1"] if prev else None; last_spk = prev["spk"] if prev else None
            gap = dur = conf = chg = spk = 0
            if t0 is not None:
                gap = 9 if last_t1 is None else (1 if t0 - last_t1 < 0 else _bk(t0 - last_t1, m["gap_edges"]))
                if t1 is not None:
                    dur = min(7, _bk((t1 - t0) / max(1, len(w["w"])), m["dur_edges"]))
            fin = 1 if w.get("final", True) else 2
            if w.get("conf") is not None:
                conf = _bk(float(w["conf"]), m["conf_edges"])
            s = w.get("spk")
            if s is not None:
                if s not in slots:
                    slots[s] = min(9, len(slots) + 1)
                spk = slots[s]; chg = 0 if last_spk is None else (1 if s == last_spk else 2)
            x = w["w"]; hd = any(ch.isdigit() for ch in x)
            if gap and t0 is not None and t1 is not None:
                fd = int(np.clip(round(t1 * 10) - round(t0 * 10) + 1, 1, 20))
                fg = 20 if (prev is None or prev["t1"] is None) else int(np.clip(round(t0 * 10) - round(prev["t1"] * 10) - 1, 0, 20))
            else:
                fd, fg = 3, (0 if prev else 20)
            prev = {"t1": t1 if t1 is not None else last_t1, "spk": s if s is not None else last_spk}
            p = self.pieces(x)
            out.append((p, [int(hd), int(hd and x.isdigit()), min(len(p), P_MAX), fd, fg],
                        [gap, dur, fin, conf, chg, asr_i, spksrc, 0], spk))
        return out

    def run(self, feats, ustart, dom):
        W = len(feats)
        pcs = np.zeros((W, P_MAX), np.int64); npc = np.zeros(W, np.int64)
        for i, (p, _, _, _) in enumerate(feats):
            pcs[i, :len(p)] = p; npc[i] = len(p)
        feed = {"pcs": pcs, "npc": npc, "wfl": np.array([f[1] for f in feats], np.int64),
                "feats": np.array([f[2] for f in feats], np.int64), "src": np.zeros(W, np.int64),
                "spk": np.array([f[3] for f in feats], np.int64), "ustart": np.array(ustart, np.int64),
                "dom": np.array([dom], np.int64)}
        return dict(zip(self.out_names, self.sess.run(self.out_names, feed)))

    # ---------------------------------------------------------------- decoding (tier T0)
    def _top(self, prof, q, P, prefix, k):
        names = self.cn[q]; th = prof["thr"][q]; va = prof["valid"][q]
        idx = [c for c, t in enumerate(names) if t.startswith(prefix) and va[c]]
        idx.sort(key=lambda c: -P[c])
        return [{"tag": names[c], "p": _r6(P[c]), "thr": _r6(th[c]), "on": bool(np.float32(P[c]) >= np.float32(th[c]))} for c in idx[:k]]

    def readouts(self, prof, O, j, spk):
        U, S, I, Fr = O["utt"][j], O["speaker"][j], O["item"][j], O["frame"][j]
        ro = {"intent": self._top(prof, "utt", U, "intent.", 3) + self._top(prof, "utt", U, "discourse.intent.", 3),
              "act": self._top(prof, "utt", U, "discourse.act.", 3) + self._top(prof, "utt", U, "discourse.answer", 1)
              + self._top(prof, "utt", U, "discourse.backchannel", 1) + self._top(prof, "utt", U, "discourse.assistant.", 1),
              "safety": [r for r in self._top(prof, "utt", U, "safety.", 20) if r["p"] >= 0.5 * r["thr"]][:6],
              "emotion": self._top(prof, "utt", U, "affect.emotion.", 3),
              "arousal": self._top(prof, "utt", U, "affect.arousal.", 3),
              "valence": self._top(prof, "utt", U, "affect.valence.", 3),
              "intensity": self._top(prof, "utt", U, "affect.intensity.", 3),
              "role": self._top(prof, "speaker", S, "speaker.role.", 3) if spk > 0 else [],
              "item": self._top(prof, "item", I, "item.", 3)}
        ro["intent"].sort(key=lambda r: -r["p"]); ro["intent"] = ro["intent"][:4]
        st = {}
        for g in self.meta["state_groups"]:
            t = self._top(prof, "frame", Fr, g + ".", 3)
            if t:
                st[g] = t
        ro["state"] = st
        return ro

    def spans(self, prof, O, us, lo, hi, a0):
        """spans over words [lo, hi) of the run (row index = word index - off handled by caller), kept when end > a0"""
        th, va = prof["thr"], prof["valid"]
        cin = self.cn["span_in"]
        idx = list(range(lo, hi))
        if not idx:
            return []
        PIN = O["span_in"][lo:hi]
        ON = (PIN >= np.array(th["span_in"])) & np.array(va["span_in"], bool)
        bo = np.array(self.meta["beg_of_in"]); hb = bo >= 0
        PB = O["span_beg"][lo:hi]
        BEG = np.zeros_like(ON)
        BEG[:, hb] = PB[:, bo[hb]] >= np.array(th["span_beg"])[bo[hb]]
        usr = np.array([bool(us[i]) for i in idx])
        BEG[usr] = True
        ci = {t: c for c, t in enumerate(cin)}
        head = {}
        for f in self.meta["span_head"]:
            if f["use"]:
                cs = [ci[t] for t in f["tags"] if t in ci and va["span_in"][ci[t]]]
                if cs:
                    head[f["name"]] = (f, cs)
        for f, cs in head.values():
            ON[:, cs] = False
        res = []
        for c in np.nonzero(ON.any(0))[0]:
            a = None
            for r in range(len(idx) + 1):
                on = r < len(idx) and ON[r, c]
                if a is not None and (not on or BEG[r, c]):
                    w0, w1 = idx[a], idx[r - 1] + 1
                    if w1 > a0:
                        res.append((int(c), w0, w1, float(PIN[a:r, c].max())))
                    a = None
                if on and a is None:
                    a = r
        res, alt = self._exclusive(prof, res, idx, PIN)
        hth = {}
        for name, (f, cols) in head.items():
            Pt = O[f["out_t"]][lo:hi]; Pb = O[f["out_b"]][lo:hi]
            tags = f["tags"]; mem = [k for k, t in enumerate(tags) if t in ci and ci[t] in cols]
            tf = prof["head_thr"][name]; pin = (np.float32(1.0) - Pb[:, 0]).astype(np.float32)
            lab = np.where(pin < np.float32(tf), 0, np.where(usr | (Pb[:, 1] > Pb[:, 2]), 1, 2))
            for r in range(1, len(idx)):
                if lab[r] == 2 and lab[r - 1] == 0:
                    lab[r] = 1
            lab[0] = 1 if lab[0] else 0
            lt = np.log(np.clip(Pt[:, mem], np.float32(1e-9), np.float32(1.0)).astype(np.float32))
            r = 0; W = len(idx)
            while r < W:
                if lab[r] == 0:
                    r += 1; continue
                e = r + 1
                while e < W and lab[e] == 2:
                    e += 1
                w0, w1 = idx[r], idx[e - 1] + 1
                if w1 > a0:
                    sc = lt[r:e].sum(0); o = np.argsort(-sc, kind="stable"); c = ci[tags[mem[int(o[0])]]]
                    res.append((c, w0, w1, float(pin[r:e].max()))); hth[c] = tf
                    if len(o) > 1:
                        pm = Pt[r:e][:, mem].mean(0)
                        alt[(c, w0, w1)] = {"tag": tags[mem[int(o[1])]], "p": _r6(pm[o[1]])}
                r = e
        out = []
        for c, a, b, p in res:
            tag = cin[c]
            s = {"type": tag, "family": family_of(tag), "start": a, "end": b, "score": _r6(p),
                 "thr": _r6(hth.get(c, th["span_in"][c])), "src": "head" if c in hth else "span_in"}
            if (c, a, b) in alt:
                s["alt"] = alt[(c, a, b)]
            if self.meta["is_norm"][c]:
                v = float(O["ver_m"][a:b].mean()) + float(O["ver_e"][b - 1])
                s["ver"] = _r6(1.0 / (1.0 + math.exp(-v)))
            out.append(s)
        out.sort(key=lambda s: (s["start"], -s["end"]))
        return out

    def _exclusive(self, prof, res, idx, PIN):
        va = prof["valid"]["span_in"]; cin = self.cn["span_in"]
        fam_cols = {}
        for c, t in enumerate(cin):
            f = decode_family(t)
            if f and va[c]:
                fam_cols.setdefault(f, []).append(c)
        row = {w: r for r, w in enumerate(idx)}
        by, keep = {}, []
        for x in res:
            f = decode_family(cin[x[0]])
            (by.setdefault(f, []) if f else keep).append(x)
        out = list(keep); alt = {}
        for f, xs in by.items():
            xs.sort(key=lambda x: (x[1], x[2])); groups = []
            for x in xs:
                if groups and x[1] < groups[-1][1]:
                    groups[-1][1] = max(groups[-1][1], x[2]); groups[-1][2].append(x)
                else:
                    groups.append([x[1], x[2], [x]])
            cols = fam_cols.get(f, [])
            for a, b, mem in groups:
                if len(mem) == 1 or not cols:
                    out.append(mem[0]); continue
                rr = [row[w] for w in range(a, b) if w in row]
                M = PIN[rr][:, cols].mean(0)
                o = np.argsort(-M, kind="stable"); c = cols[int(o[0])]
                out.append((c, a, b, float(PIN[rr, c].max())))
                if len(o) > 1:
                    alt[(c, a, b)] = {"tag": cin[cols[int(o[1])]], "p": _r6(M[o[1]])}
        return out, alt

    def events(self, prof, O, words, keys, lo, hi, off=0):
        """T0 events of words [lo, hi) (global indices; rows = index - off); keys: (tag, word) already emitted"""
        out = []
        for i in range(lo, hi):
            for q in ("ev_at", "ev_imm", "ev_rec"):
                p = O[q][i - off]
                if not len(p):
                    continue
                on = np.nonzero((p >= np.array(prof["thr"][q])) & np.array(prof["valid"][q], bool))[0]
                for c in on:
                    tag = self.cn[q][c]
                    if tag in self.ev_skip:
                        continue
                    if any((tag, j) in keys for j in range(i - 3, i + 1)):
                        continue
                    keys.add((tag, i))
                    out.append({"type": tag, "q": q, "word": i, "score": _r6(p[c]), "thr": _r6(prof["thr"][q][c]), "text": words[i]["w"]})
        return out

    # ---------------------------------------------------------------- API
    def segments(self, key, words, ustart):
        """fine-domain switches (sports / court): after each utterance, whenever len(words) // fine_every
        changed, the lexicon cues of the words so far may re-condition the session on another fine domain of the same UI
        domain. Returns [(word index the profile starts at, profile key)]; the utterance ending at a switch point was read
        before the switch."""
        prof = self.meta["profiles"][key]; ui = prof["ui"]
        segs = [(0, key)]
        if not self.fine_auto or ui not in (self.meta.get("fine_cues") or {}):
            return segs
        n = len(words); fe = self.meta.get("fine_every", 25); chk = -1; cur = prof["fine"]
        for e in [i for i in range(1, n + 1) if i == n or ustart[i]]:
            if e // fe != chk:
                chk = e // fe
                fh = self.fine_hint(ui, words[:e])
                if fh and fh != cur and f"{ui}:{fh}" in self.meta["profiles"]:
                    segs.append((e, f"{ui}:{fh}")); cur = fh
        return segs

    def tag_words(self, words, ustart, domain="auto", asr=None):
        key, _ = self.profile(domain)
        n = len(words); mc = self.meta["max_ctx"]; sw = self.meta["span_win"]
        segs = self.segments(key, words, ustart)
        profs = [self._thresholds(self.meta["profiles"][k]) for _, k in segs]
        sw_at = [s for s, _ in segs] + [n + 1]
        cut = [-1] + [max(0, s - sw) for s, _ in segs[1:]] + [n + 1]      # spans ending in (cut[k], cut[k+1]] come from segment k
        res = {"profile": segs[-1][1], "n_words": n, "words": [w["w"] for w in words], "spans": [], "events": [], "utterances": []}
        keys = set()
        for c0 in range(0, n, mc):
            c1 = min(n, c0 + mc)
            ws = words[c0:c1]; us = list(ustart[c0:c1]); us[0] = 1
            feats = self.features(ws, asr)
            Os, sp = [], []
            for k, prof in enumerate(profs):
                O = self.run(feats, us, prof["dom"]); Os.append(O)
                for s in self.spans(prof, O, us, 0, c1 - c0, -1):
                    s["start"] += c0; s["end"] += c0
                    if cut[k] < s["end"] <= cut[k + 1]:
                        s["text"] = " ".join(w["w"] for w in words[s["start"]:s["end"]])
                        sp.append(s)
            sp.sort(key=lambda s: (s["start"], -s["end"]))
            res["spans"] += sp
            for k, prof in enumerate(profs):
                res["events"] += self.events(prof, Os[k], words, keys, c0, min(c1, sw_at[k + 1]), c0)
            starts = [i for i in range(c1 - c0) if us[i]] + [c1 - c0]
            for a, b in zip(starts[:-1], starts[1:]):
                k = max(j for j, (s0, _) in enumerate(segs) if s0 < b + c0 or j == 0)
                u = {"start": a + c0, "end": b + c0, "text": " ".join(w["w"] for w in ws[a:b])}
                u.update(self.readouts(profs[k], Os[k], b - 1, feats[b - 1][3]))
                res["utterances"].append(u)
        return self.finish(res, words)

    # ---------------------------------------------------------------- redaction + output filters
    def redacted_text(self, words, spans):
        pol = self.redact
        lab = [None] * len(words)
        for x in sorted(spans, key=lambda x: (x["start"], -x["end"], x["type"])):
            c = REDACT_CAT.get(x["type"])
            if c and pol.get(c) == "mask":
                for i in range(x["start"], min(x["end"], len(words))):
                    lab[i] = lab[i] or (REDACT_LABEL[c], x["start"])
        out, i = [], 0
        while i < len(words):
            if lab[i] is None:
                out.append(words[i]["w"]); i += 1; continue
            j = i
            while j < len(words) and lab[j] == lab[i]:
                j += 1
            out.append(f"[{lab[i][0]}]"); i = j
        return " ".join(out)

    def _keep(self, tag):
        return self.tags is None or any(tag.startswith(p) for p in self.tags)

    def finish(self, res, words):
        if self.redact is not None:
            res["redacted_text"] = self.redacted_text(words, res["spans"])
        if self.emit is None and self.tags is None:
            return res
        em = self.emit if self.emit is not None else set(SPAN_SECTIONS) | {"events"} | set(READOUT_KEYS)
        res["spans"] = [s for s in res["spans"] if span_section(s["type"]) in em and self._keep(s["type"])]
        res["events"] = [e for e in res["events"] if "events" in em and self._keep(e["type"])] if "events" in em else []
        for u in res["utterances"]:
            for k in READOUT_KEYS:
                if k not in em:
                    u.pop(k, None)
                elif k == "state":
                    u[k] = {g: [r for r in v if self._keep(r["tag"])] for g, v in u[k].items()}
                    u[k] = {g: v for g, v in u[k].items() if v}
                else:
                    u[k] = [r for r in u[k] if self._keep(r["tag"])]
        return res

    def tag_text(self, text, domain="auto", asr=None):
        ws = split_words(text)
        return self.tag_words([{"w": w} for w in ws], utt_starts(ws), domain, asr)


if __name__ == "__main__":
    md = sys.argv[1]
    if sys.argv[2] == "--tokens":
        t = ArcTagger(md)
        print(json.dumps([t.tok.encode(" " + w.strip(), add_special_tokens=False).ids for w in split_words(sys.argv[3])]))
    else:
        print(json.dumps(ArcTagger(md).tag_text(sys.argv[2], *(sys.argv[3:5])), indent=1))
