"""Python reference for the arctag C library (numpy + onnxruntime + tokenizers).

Same input features, model and decoding as the C library (and as the Arc Tagger demo engine's T0 path):
  text -> whitespace words -> utterances (a word ending in . ? ! closes one) -> per-word features -> arctag.onnx ->
  spans (span head per family + span_in runs), events (ev_at / ev_imm / ev_rec, 3-word de-dup), utterance read-outs
  (intent / act / safety / emotion / arousal / item / state / role) with the per-profile display thresholds of arctag_meta.json.

  python arctag_ref.py MODEL_DIR "some text"            -> JSON
  python arctag_ref.py MODEL_DIR --tokens "some text"   -> token ids per word
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
    def __init__(self, model_dir, threads=1):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.meta = json.load(open(os.path.join(model_dir, "arctag_meta.json")))
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(model_dir, "arctag.onnx"), so, providers=["CPUExecutionProvider"])
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.cn = self.meta["readouts"]
        self.asr_sources = self.meta["asr_sources"]

    # ---------------------------------------------------------------- inputs
    def pieces(self, w):
        ids = self.tok.encode(" " + w.strip(), add_special_tokens=False).ids or [0]
        return ids[:P_MAX - 1] + ids[-1:] if len(ids) > P_MAX else ids

    def profile(self, domain):
        m = self.meta
        key = domain if domain in m["profiles"] else m["aliases"].get(domain or "auto", m["default_profile"])
        return key, m["profiles"][key]

    def features(self, words, asr):
        """words: dicts {w, t0?, t1?, conf?, spk?, final?} -> per-word (pieces, wfl, feats, spk slot) like the engine"""
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

    # ---------------------------------------------------------------- decoding (= demo engine ArcSession, tier T0)
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
                    if any((tag, j) in keys for j in range(i - 3, i + 1)):
                        continue
                    keys.add((tag, i))
                    out.append({"type": tag, "q": q, "word": i, "score": _r6(p[c]), "thr": _r6(prof["thr"][q][c]), "text": words[i]["w"]})
        return out

    # ---------------------------------------------------------------- API
    def tag_words(self, words, ustart, domain="auto", asr=None):
        key, prof = self.profile(domain)
        n = len(words); mc = self.meta["max_ctx"]
        res = {"profile": key, "n_words": n, "words": [w["w"] for w in words], "spans": [], "events": [], "utterances": []}
        keys = set()
        for c0 in range(0, n, mc):
            c1 = min(n, c0 + mc)
            ws = words[c0:c1]; us = list(ustart[c0:c1]); us[0] = 1
            feats = self.features(ws, asr)
            O = self.run(feats, us, prof["dom"])
            sp = self.spans(prof, O, us, 0, c1 - c0, -1)
            for s in sp:
                s["start"] += c0; s["end"] += c0
                s["text"] = " ".join(w["w"] for w in words[s["start"]:s["end"]])
            res["spans"] += sp
            res["events"] += self.events(prof, O, words, keys, c0, c1, c0)
            starts = [i for i in range(c1 - c0) if us[i]] + [c1 - c0]
            for a, b in zip(starts[:-1], starts[1:]):
                u = {"start": a + c0, "end": b + c0, "text": " ".join(w["w"] for w in ws[a:b])}
                u.update(self.readouts(prof, O, b - 1, feats[b - 1][3]))
                res["utterances"].append(u)
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
