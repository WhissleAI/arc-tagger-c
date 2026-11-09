#!/usr/bin/env python3
"""tools/export_model.py on altered graphs: a new checkpoint of the same shape must export, with generated names changed, and anything
else must fail with a clear message and leave no weights file behind. usage: test_export.py MODEL_DIR TMP_DIR

Each case writes one ONNX file of about 35 MB into TMP_DIR and removes it afterwards."""
import os
import shutil
import subprocess
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper as nh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(ROOT, "tools", "export_model.py")


def run(src, out, *extra):
    return subprocess.run([sys.executable, EXPORT, src, "--out", out, *extra], capture_output=True, text=True)


def replace_init(g, name, arr):
    for i, t in enumerate(g.initializer):
        if t.name == name:
            g.initializer.remove(t)
            g.initializer.insert(i, nh.from_array(arr, name))
            return
    raise KeyError(name)


def new_checkpoint(m):
    """same graph, other weights, every generated name changed: what a new training run looks like"""
    rng = np.random.default_rng(7)
    g = m.graph
    ren = {}
    for t in list(g.initializer):
        a = nh.to_array(t)
        if a.dtype == np.float32:
            replace_init(g, t.name, (a * (1 + 0.02 * rng.standard_normal(a.shape))).astype(np.float32))
    for i, t in enumerate(g.initializer):
        if t.name.startswith("onnx::"):
            ren[t.name] = f"onnx::Mul_{9000 + i}"
            t.name = ren[t.name]
    for n in g.node:
        for k, x in enumerate(n.input):
            if x in ren:
                n.input[k] = ren[x]


def wider_table(m):
    a = next(t for t in m.graph.initializer if t.name == "core.L0_Wf")
    replace_init(m.graph, "core.L0_Wf", np.zeros((a.dims[0], a.dims[1] + 1), np.float32))


def extra_parameter(m):
    m.graph.initializer.append(nh.from_array(np.zeros(256, np.float32), "core.L6_nw"))


def anonymous_gone(m):
    """the distance slope is no longer a constant"""
    t = next(t for t in m.graph.initializer if t.name.startswith("onnx::Mul_"))
    m.graph.initializer.remove(t)
    m.graph.node.insert(0, helper.make_node("Neg", [t.name + "_x"], [t.name]))
    m.graph.initializer.append(nh.from_array(nh.to_array(t), t.name + "_x"))


def one_einsum_fewer(m):
    n = next(n for n in m.graph.node if n.op_type == "Einsum")
    m.graph.node.remove(n)


def changed_temperature(m):
    """passes every structural check, changes the numbers: only the parity check can see it"""
    g = m.graph
    prods = {o: n for n in g.node for o in n.output}
    for n in g.node:
        if n.op_type == "Div" and prods.get(n.input[0]) is not None and prods[n.input[0]].op_type == "Einsum":
            c = prods.get(n.input[1])
            if c is not None and c.op_type == "Constant":
                for at in c.attribute:
                    if at.name == "value":
                        at.t.CopyFrom(nh.from_array(np.array(4.0, np.float32), at.t.name))
                return
    raise RuntimeError("no Div(Einsum, const) in the graph")


CASES = [
    ("new weights, new generated names", new_checkpoint, 0, "parity passed"),
    ("a table of another width", wider_table, 1, "graph shape mismatch"),
    ("a parameter the runtime does not know", extra_parameter, 1, "does not know"),
    ("an anonymous constant that became computed", anonymous_gone, 1, "graph shape mismatch"),
    ("an operator fewer", one_einsum_fewer, 1, "graph shape mismatch"),
    ("same structure, different arithmetic", changed_temperature, 1, "parity check FAILED"),
]


def main():
    src, tmp = sys.argv[1:3]
    os.makedirs(tmp, exist_ok=True)
    work = os.path.join(tmp, "src")
    bad = 0
    for name, fn, want_rc, needle in CASES:
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        for f in ("arctag_meta.json", "tokenizer.json"):
            shutil.copyfile(os.path.join(src, f), os.path.join(work, f))
        m = onnx.load(os.path.join(src, "arctag.onnx"))
        fn(m)
        onnx.save(m, os.path.join(work, "arctag.onnx"))
        out = os.path.join(tmp, "out")
        shutil.rmtree(out, ignore_errors=True)
        r = run(work, out, "--weights", "fp32")
        text = r.stdout + r.stderr
        ok = (r.returncode != 0) == bool(want_rc) and needle in text
        wrote = os.path.exists(os.path.join(out, "arctag.atw"))
        if want_rc and wrote:
            ok = False
        last = [x for x in text.strip().splitlines() if "ERROR" in x or "exported" in x]
        print(f"  {'ok ' if ok else 'BAD'} {name}: exit {r.returncode}, {last[0][:150] if last else ''}")
        if not ok:
            print(text[-2000:])
            bad += 1
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([sys.executable, EXPORT, os.path.join(tmp, "nothing_here")], capture_output=True, text=True)
    if r.returncode == 0 or "has no arctag.onnx" not in r.stderr:
        print("  BAD a missing source:", r.stderr)
        bad += 1
    print(f"test_export: {len(CASES) + 1} cases, {bad} wrong")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
