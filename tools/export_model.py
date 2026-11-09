#!/usr/bin/env python3
"""Export an Arc Tagger ONNX graph to the native runtime's weights file (arctag.atw), and prove the result is right.

  python3 tools/export_model.py SRC_DIR --out OUT_DIR --weights int8

SRC_DIR holds arctag.onnx, arctag_meta.json and tokenizer.json. OUT_DIR receives arctag.atw and copies of the other two, ready
for arctag_create. --weights: int8 (quantize first, with tools/quantize_model.py; what the apps ship), fp32, or asis (default:
store the tables exactly as the ONNX holds them). --keep-onnx also leaves the matching arctag.onnx in OUT_DIR, for tests.

Nothing is written to OUT_DIR unless every step passes. The exit status is non-zero, with the reason on stderr, if:
  1. the graph's inputs, outputs, tensors or layer structure are not the architecture src/atn.c implements;
  2. a tensor in the graph would go unused by the runtime (a new parameter must not be dropped silently);
  3. an anonymous constant cannot be found by its place in the graph. Nothing here depends on generated names like
     onnx::Mul_3635; such values are found by walking from named parameters;
  4. the native forward pass, built from this checkout's src/atn.c, differs from ONNX Runtime on the same graph by more than
     --tol (probabilities) or --tol-logit (the verifier logits), over the reference texts in tools/make_refs.py.

Step 4 is the one that matters: steps 1-3 give a clear message, step 4 catches whatever they could not imagine.
Needs: numpy, onnx, onnxruntime, tokenizers, and a C compiler.
"""
import argparse
import collections
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import onnx
from onnx import numpy_helper as nh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

D, HH, DH, NST, PROJ = 256, 4, 64, 64, 1160
NENC, NLAY = 2, 6
READOUTS = ["frame", "span_beg", "span_in", "span_end", "ev_at", "ev_imm", "ev_rec", "utt", "speaker", "item"]
SPAN_HEAD = ["sh_entity_number_t", "sh_entity_number_b", "sh_entity_mention_t", "sh_entity_mention_b", "sh_safety_pii_t",
             "sh_safety_pii_b", "sh_safety_profanity_t", "sh_safety_profanity_b", "sh_safety_toxicity_t", "sh_safety_toxicity_b"]
OUTPUTS = READOUTS + SPAN_HEAD + ["ver_m", "ver_e"]
INPUTS = ["pcs", "npc", "wfl", "feats", "src", "spk", "ustart", "dom"]
PROJ_SPLIT = [D, D, D, D, 128, 4, 4]
# Operators whose count is a property of the architecture, not of the tag set. A different count means the forward pass in
# src/atn.c no longer describes the graph (read-out group counts move Softmax, Concat, Gather and Slice, so those are not listed).
ARCH_OPS = {"Einsum": 20, "Erf": 12, "Softplus": 6, "Split": 6, "Log": 1, "Sin": 8, "Cos": 8, "CumSum": 18, "Exp": 7, "Pow": 43, "Sqrt": 43}

# file format ATW2: must match src/atn.c
ATW_VERSION, ATW_ENDIAN, BLOB_ALIGN, BLOB_FILE_ALIGN = 2, 0x01020304, 64, 4096
DT = {"f": 0, "i8": 1, "i32": 2, "u8": 3}


class ExportError(Exception):
    pass


def fail(msg):
    raise ExportError(msg)


def expected_tensors():
    """name -> shape, None for 'any size' (those are checked for consistency below). Mirrors load_tensors in src/atn.c."""
    t = {}
    for n, s in [("tokE", (None, D)), ("W1", (D, D)), ("W2", (D, D)), ("src", (None, D)), ("ppos", (8, D)), ("b1", (D,)), ("lnw", (D,)), ("lnb", (D,)),
                 ("rope_enc", (DH // 2,)), ("encnw", (D,)), ("encnb", (D,)), ("in_nw", (D,)), ("in_nb", (D,)), ("Winp", (D, D)), ("binp", (None, D)),
                 ("adD", (None, D, 16)), ("adU", (None, 16, D)), ("onw", (D,)), ("dq0", (D,)), ("mr_nw", (D,)), ("mr_q", (D, DH)), ("mr_k", (D, DH)), ("mr_v", (D, DH)),
                 ("mr_qnw", (DH,)), ("mr_knw", (DH,)), ("mr_kmW", (D,)), ("mr_kmb", (1,)), ("mr_o", (DH, D)), ("mr_ou", (DH, None)), ("mr_kb", (1,)), ("valid", (None, None)),
                 ("war_rb", (D,)), ("war_k0", (D,)), ("war_lnw", (D,)), ("war_lnb", (D,)), ("war_M", (D, None)), ("war_b", (None,)), ("shW", (D, None)), ("shb", (None,)),
                 ("shd", (len(SPAN_HEAD), 2)), ("ua_w", (D,)), ("ua_b", (1,)), ("ver_w1", (D,)), ("ver_w2", (D,)), ("ver_b", (1,))]:
        t[n] = s
    for i in range(5):
        t[f"fl{i}"] = (None, D)
    for n in ["gap", "dur", "fin", "conf", "chg", "asr", "spksrc", "turn"]:
        t[f"feat_{n}"] = (None, D)
    for l in range(NENC):
        for n, s in [("n1w", (D,)), ("n1b", (D,)), ("Wqkv", (D, 3 * D)), ("bqkv", (3 * D,)), ("Wo", (D, D)), ("bo", (D,)), ("n2w", (D,)), ("n2b", (D,)),
                     ("F1", (D, 2 * D)), ("f1", (2 * D,)), ("F2", (2 * D, D)), ("f2", (D,))]:
            t[f"enc{l}_{n}"] = s
    for l in range(NLAY):
        for n, s in [("g", (None, D)), ("b", (None, D)), ("Wf", (D, PROJ)), ("bf", (PROJ,)), ("nw", (D,)), ("A", (HH,)), ("Bw", (NST,)), ("Cw", (NST,)),
                     ("Bb", (HH, NST)), ("Cb", (HH, NST)), ("Wout", (D, D)), ("bout", (D,)), ("mnw", (D,)), ("M1", (D, 2 * D)), ("m1", (2 * D,)), ("M2", (2 * D, D)), ("m2", (D,))]:
            t[f"L{l}_{n}"] = s
    for l in (1, 3):
        for n, s in [("nw", (D,)), ("Wqkv", (D, 3 * D)), ("qnw", (DH,)), ("knw", (DH,)), ("Wo", (D, D)), ("kmW", (D,)), ("kmb", (1,)), ("slope", (HH,)), ("kb", (HH,))]:
            t[f"G{l}_{n}"] = s
        t[f"rope_g{l}"] = (DH // 2,)
    for p in ["utt", "speaker", "item"]:
        t[f"P1_{p}"] = (D, D); t[f"p1_{p}"] = (D,); t[f"P2_{p}"] = (D, D); t[f"p2_{p}"] = (D,)
    for q in READOUTS:
        if q != "span_beg":
            t[f"M_{q}"] = (D, None); t[f"b_{q}"] = (None,)
        t[f"c_{q}"] = (None,); t[f"gp_{q}"] = (None,); t[f"gi_{q}"] = (None,)
    return t


class Graph:
    def __init__(self, path):
        self.model = onnx.load(path)
        self.g = self.model.graph
        self.nodes = [n for n in self.g.node if n.op_type != "Constant"]
        self.index = {id(n): i for i, n in enumerate(self.nodes)}
        self.init = {t.name: nh.to_array(t) for t in self.g.initializer}
        self.const = {}
        for n in self.g.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        self.const[n.output[0]] = nh.to_array(a.t)
        self.alias = {n.output[0]: n.input[0] for n in self.nodes if n.op_type == "Identity"}
        self.consumers = collections.defaultdict(list)
        for n in self.nodes:
            for i in n.input:
                self.consumers[i].append(n)
        self.producer = {o: n for n in self.nodes for o in n.output}

    def value(self, name):
        """the array behind a name, through Identity; None if it is computed"""
        while name in self.alias:
            name = self.alias[name]
        if name in self.init:
            return self.init[name]
        return self.const.get(name)

    def only_consumer(self, name, op, why):
        c = [n for n in self.consumers[name] if n.op_type == op]
        if len(c) != 1:
            fail(f"graph shape mismatch: {why}: expected one {op} using {name}, found {len(c)}")
        return c[0]

    def const_input(self, node, shape, why):
        """the single constant input of node with this shape"""
        hits = [i for i in node.input if (v := self.value(i)) is not None and v.shape == shape]
        if len(hits) != 1:
            fail(f"graph shape mismatch: {why}: {node.op_type} node has {len(hits)} constant inputs of shape {shape}")
        return hits[0], self.value(hits[0])


def collect_tensors(G):
    """Every parameter under its plain name: initializers, DequantizeLinear (int8 + per-column scale) and Identity aliases."""
    strip = lambda s: s[5:] if s.startswith("core.") else s
    T, used = {}, set()
    dq = {n.output[0]: n for n in G.nodes if n.op_type == "DequantizeLinear"}
    for out, n in dq.items():
        q, s, z = n.input
        for x in (q, s, z):
            if x not in G.init:
                fail(f"graph shape mismatch: DequantizeLinear {out} has a computed input {x}")
        if np.any(G.init[z] != 0):
            fail(f"table {out}: non-zero zero points are not supported")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        if G.init[q].dtype != np.int8 or attrs.get("axis") != 1 or G.init[q].ndim != 2:
            fail(f"table {out}: only int8 per-column (axis=1) 2-D tables are supported")
        T[strip(out)] = (G.init[q], G.init[s].astype(np.float32))
        used.update((q, s, z))
    for name, arr in G.init.items():
        if name not in used:
            T[strip(name)] = (arr, None)
    for out, src in G.alias.items():
        s = strip(src)
        if s not in T:
            fail(f"graph shape mismatch: Identity {out} of unknown tensor {src}")
        T[strip(out)] = T[s]
    return T


def find_anonymous(G, T):
    """Values the exporter of the graph left without a stable name, found by where they sit relative to named parameters."""
    anon = {}
    # global attention, biases on the score: Add(Mul(slope, distance), Mul(kb, Unsqueeze(Sigmoid(Add(MatMul(x, kmW), kmb)))))
    for li in (1, 3):
        why = f"global attention after layer {li}"
        kmb = f"core.G{li}_kmb"
        if kmb not in G.consumers:
            fail(f"graph shape mismatch: {why}: {kmb} is not used")
        add = G.only_consumer(kmb, "Add", why)
        sig = G.only_consumer(add.output[0], "Sigmoid", why)
        unsq = G.only_consumer(sig.output[0], "Unsqueeze", why)
        mul_kb = G.only_consumer(unsq.output[0], "Mul", why)
        kb_name, kb = G.const_input(mul_kb, (HH, 1), why + ", score bias")
        add2 = G.only_consumer(mul_kb.output[0], "Add", why)
        others = [i for i in add2.input if i != mul_kb.output[0]]
        mul_slope = G.producer.get(others[0]) if len(others) == 1 else None
        if mul_slope is None or mul_slope.op_type != "Mul":
            fail(f"graph shape mismatch: {why}: the distance term is not a Mul")
        slope_name, slope = G.const_input(mul_slope, (HH, 1), why + ", distance slope")
        anon[f"G{li}_slope"] = slope[:, 0].astype(np.float32)
        anon[f"G{li}_kb"] = kb[:, 0].astype(np.float32)
        G.consumed_anon.update((kb_name, slope_name))
    # RoPE frequencies: Mul(position, const [1, 32]) feeding Cos and Sin; one table for local attention, one for global attention
    rope = []
    for n in G.nodes:
        if n.op_type == "Mul":
            for i in n.input:
                v = G.value(i)
                if v is not None and v.shape == (1, DH // 2):
                    rope.append((G.index[id(n)], i, v))
    if len(rope) != 2:
        fail(f"graph shape mismatch: expected 2 RoPE frequency tables [1, {DH // 2}], found {len(rope)}")
    use = lambda name: min(G.index[id(n)] for n in G.consumers[name])
    enc_end, g_end = use("core.enc1_Wo"), use("core.G1_Wo")
    if not (rope[0][0] < enc_end < rope[1][0] < g_end):
        fail("graph shape mismatch: RoPE tables are not one before the local and one before the global attention")
    anon["rope_enc"] = rope[0][2][0].astype(np.float32)
    anon["rope_g1"] = anon["rope_g3"] = rope[1][2][0].astype(np.float32)
    G.consumed_anon.update((rope[0][1], rope[1][1]))
    # mention memory prior: Mul(Log(Sigmoid(Add(MatMul(x, kmW), kmb))), kb)
    why = "mention memory prior"
    add = G.only_consumer("core.mr_kmb", "Add", why)
    sig = G.only_consumer(add.output[0], "Sigmoid", why)
    log = G.only_consumer(sig.output[0], "Log", why)
    mul = G.only_consumer(log.output[0], "Mul", why)
    kb_name, kb = G.const_input(mul, (), why)
    anon["mr_kb"] = np.array([float(kb)], np.float32)
    G.consumed_anon.add(kb_name)
    # span head: each output is Softmax(Slice(shW output, start, end)). The sizes follow the checkpoint's tag set.
    outs = {n.output[0]: n for n in G.nodes}
    shd = []
    for name in SPAN_HEAD:
        sm = outs.get(name)
        if sm is None or sm.op_type != "Softmax":
            fail(f"graph shape mismatch: output {name} is not a Softmax")
        sl = G.producer.get(sm.input[0])
        if sl is None or sl.op_type != "Slice":
            fail(f"graph shape mismatch: span head output {name} is not a Slice of the head")
        a, b = G.value(sl.input[1]), G.value(sl.input[2])
        if a is None or b is None:
            fail(f"graph shape mismatch: span head output {name} has computed slice bounds")
        shd.append((int(a.ravel()[0]), int(b.ravel()[0])))
    anon["shd"] = np.array(shd, np.int32)
    return anon


def check_structure(G, T):
    errs = []
    ins = [i.name for i in G.g.input]
    outs = [o.name for o in G.g.output]
    if ins != INPUTS:
        errs.append(f"inputs are {ins}, expected {INPUTS}")
    if outs != OUTPUTS:
        errs.append(f"outputs are {outs}, expected {OUTPUTS}")
    counts = collections.Counter(n.op_type for n in G.nodes)
    for op, want in ARCH_OPS.items():
        if counts[op] != want:
            errs.append(f"{counts[op]} {op} operators, expected {want}")
    for n in G.nodes:
        if n.op_type == "Split":
            v = G.value(n.input[1]) if len(n.input) > 1 else None
            if v is None or list(v) != PROJ_SPLIT:
                errs.append(f"trunk projection split is {None if v is None else list(v)}, expected {PROJ_SPLIT}")
                break
    want = expected_tensors()
    missing = sorted(k for k in want if k not in T)
    if missing:
        errs.append(f"missing tensors: {missing}")
    known = lambda k: k in want or re.fullmatch(r"gg_(%s)_\d+" % "|".join(READOUTS), k)
    extra = sorted(k for k in T if not known(k))
    if extra:
        errs.append(f"tensors the runtime does not know (a new parameter or layer?): {extra}")
    for k, shape in want.items():
        if k not in T:
            continue
        got = T[k][0].shape
        if len(got) != len(shape) or any(s is not None and s != g for s, g in zip(shape, got)):
            errs.append(f"tensor {k} has shape {got}, expected {shape} (None = any)")
    if errs:
        return errs
    # the free sizes must agree with each other
    nd = T["binp"][0].shape[0]
    for k in ["adD", "adU", "valid"] + [f"L{l}_{x}" for l in range(NLAY) for x in "gb"]:
        if T[k][0].shape[0] != nd:
            errs.append(f"tensor {k} has {T[k][0].shape[0]} domains, binp has {nd}")
    for q in READOUTS:
        M = "war_M" if q == "span_beg" else f"M_{q}"
        b = "war_b" if q == "span_beg" else f"b_{q}"
        n = T[M][0].shape[1]
        if T[b][0].shape[0] != n or T[f"c_{q}"][0].shape[0] != n or T[f"gp_{q}"][0].shape[0] != n:
            errs.append(f"read-out {q}: {M} has {n} columns but {b}, c_{q} or gp_{q} differ")
    if T["mr_ou"][0].shape[1] != T["war_M"][0].shape[1]:
        errs.append("mr_ou and war_M have different column counts")
    sh = T["shd"][0]
    at = 0
    for f, (a, b) in enumerate(sh):
        if a != at or b <= a:
            errs.append(f"span head slice {SPAN_HEAD[f]} is [{a}, {b}), expected it to start at {at}")
        at = b
    if at != T["shW"][0].shape[1] or T["shb"][0].shape[0] != at:
        errs.append(f"span head slices cover {at} columns; shW has {T['shW'][0].shape[1]}")
    return errs


def write_atw(T, kind, source_sha, dst):
    names = sorted(T)
    blob = bytearray()
    ents = []

    def put(arr):
        blob.extend(b"\0" * (-len(blob) % BLOB_ALIGN))
        off = len(blob)
        blob.extend(arr.tobytes())
        return off

    for nm in names:
        a, s = T[nm]
        if a.dtype == np.int64:
            if a.size and (a.min() < -2**31 or a.max() >= 2**31):
                fail(f"tensor {nm}: values do not fit 32 bits")
            a = a.astype(np.int32)
        elif a.dtype == np.bool_:
            a = a.astype(np.uint8)
        dtc = {np.dtype("float32"): DT["f"], np.dtype("int8"): DT["i8"], np.dtype("int32"): DT["i32"], np.dtype("uint8"): DT["u8"]}.get(a.dtype)
        if dtc is None:
            fail(f"tensor {nm}: unsupported type {a.dtype}")
        enc = nm.encode()
        if not 0 < len(enc) < 48 or a.ndim > 4:
            fail(f"tensor {nm}: name or rank does not fit the file format")
        off = put(a)
        soff = put(s.astype(np.float32)) if s is not None else -1
        shape = list(a.shape) + [0] * (4 - a.ndim)
        ents.append(enc.ljust(48, b"\0") + struct.pack("<6i2q", dtc, a.ndim, *shape, off, soff))
    table = b"".join(ents)
    blob_off = 64 + len(table)
    blob_off += -blob_off % BLOB_FILE_ALIGN
    payload = table + b"\0" * (blob_off - 64 - len(table)) + bytes(blob)
    file_size = 64 + len(payload)
    head = b"ATW2" + struct.pack("<3I2Q", ATW_VERSION, ATW_ENDIAN, len(names), file_size, blob_off)
    head += kind.encode().ljust(8, b"\0") + source_sha[:8] + b"\0" * 8 + struct.pack("<I", zlib.crc32(payload))
    head += struct.pack("<I", zlib.crc32(head))
    assert len(head) == 64
    with open(dst, "wb") as f:
        f.write(head)
        f.write(payload)


def build_checker(tmp):
    cc = os.environ.get("CC", "cc")
    exe = os.path.join(tmp, "test_native_parity")
    cmd = [cc, "-O2", "-I" + os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "atn.c"), os.path.join(ROOT, "tests", "test_native_parity.c"), "-lm", "-o", exe]
    if sys.platform == "darwin":
        cmd[1:1] = ["-framework", "Accelerate"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        fail(f"cannot build the parity checker:\n{' '.join(cmd)}\n{r.stderr}")
    return exe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="directory with arctag.onnx, arctag_meta.json, tokenizer.json")
    ap.add_argument("--out", help="output directory (default: SRC)")
    ap.add_argument("--weights", choices=["asis", "fp32", "int8"], default="asis")
    ap.add_argument("--tol", type=float, default=2e-4, help="largest allowed difference on probabilities")
    ap.add_argument("--tol-logit", type=float, default=5e-4, help="largest allowed difference on the verifier logits")
    ap.add_argument("--keep-onnx", action="store_true", help="also leave the matching arctag.onnx in OUT")
    a = ap.parse_args()
    out = a.out or a.src
    onnx_in = os.path.join(a.src, "arctag.onnx")
    for f in ["arctag.onnx", "arctag_meta.json", "tokenizer.json"]:
        if not os.path.isfile(os.path.join(a.src, f)):
            fail(f"{a.src} has no {f}")
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="atw_", dir=out) as tmp:
        work = os.path.join(tmp, "model")
        os.makedirs(work)
        for f in ["arctag_meta.json", "tokenizer.json"]:
            shutil.copyfile(os.path.join(a.src, f), os.path.join(work, f))
        onnx_work = os.path.join(work, "arctag.onnx")
        has_q = any(n.op_type == "DequantizeLinear" for n in onnx.load(onnx_in, load_external_data=False).graph.node)
        if a.weights == "fp32" and has_q:
            fail(f"--weights fp32 but {onnx_in} is already quantized")
        if a.weights == "int8" and not has_q:
            r = subprocess.run([sys.executable, os.path.join(HERE, "quantize_model.py"), onnx_in, onnx_work], capture_output=True, text=True)
            if r.returncode:
                fail(f"quantize_model.py failed:\n{r.stdout}{r.stderr}")
            print(r.stdout.strip())
        else:
            shutil.copyfile(onnx_in, onnx_work)
        G = Graph(onnx_work)
        G.consumed_anon = set()
        T = collect_tensors(G)
        for k, v in find_anonymous(G, T).items():
            T[k] = (v, None)
        anon_left = sorted(k for k in T if k.startswith("onnx::") and k not in G.consumed_anon)
        if anon_left:
            fail("graph shape mismatch: anonymous parameters the runtime does not know about: " + ", ".join(anon_left))
        T = {k: v for k, v in T.items() if not k.startswith("onnx::")}
        errs = check_structure(G, T)
        if errs:
            fail("graph shape mismatch: this graph is not the architecture src/atn.c implements:\n  - " + "\n  - ".join(errs))
        print(f"structure ok: {len(T)} tensors, {T['binp'][0].shape[0]} domains, read-outs "
              + " ".join(f"{q}={(T['war_M'] if q == 'span_beg' else T['M_' + q])[0].shape[1]}" for q in READOUTS))
        kind = "int8" if any(s is not None for _, s in T.values()) else "fp32"
        sha = hashlib.sha256(open(onnx_work, "rb").read()).digest()
        atw = os.path.join(work, "arctag.atw")
        write_atw(T, kind, sha, atw)
        print(f"wrote {kind} weights: {os.path.getsize(atw) / 1e6:.1f} MB; source sha256 {sha.hex()[:16]}")
        # parity: ONNX Runtime on this graph vs the native forward pass on this file
        refs = os.path.join(tmp, "refs.bin")
        r = subprocess.run([sys.executable, os.path.join(HERE, "make_refs.py"), work, refs], capture_output=True, text=True)
        if r.returncode:
            fail(f"make_refs.py failed:\n{r.stdout}{r.stderr}")
        print(r.stdout.strip())
        exe = build_checker(tmp)
        r = subprocess.run([exe, atw, refs, str(a.tol), str(a.tol_logit)], capture_output=True, text=True)
        print(r.stdout.rstrip())
        if r.returncode:
            fail(f"parity check FAILED against {onnx_in} (exit {r.returncode}). The file was not written.\n{r.stderr}")
        # all good: publish
        for f in ["arctag_meta.json", "tokenizer.json"]:
            if os.path.abspath(out) != os.path.abspath(a.src):
                shutil.copyfile(os.path.join(work, f), os.path.join(out, f))
        if a.keep_onnx and os.path.abspath(out) != os.path.abspath(a.src):
            shutil.copyfile(onnx_work, os.path.join(out, "arctag.onnx"))
        os.replace(atw, os.path.join(out, "arctag.atw"))
    print(f"exported {os.path.join(out, 'arctag.atw')} ({kind}); parity passed")


if __name__ == "__main__":
    try:
        main()
    except ExportError as e:
        print(f"export_model: ERROR: {e}", file=sys.stderr)
        sys.exit(1)
