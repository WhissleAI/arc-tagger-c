#!/usr/bin/env python3
"""Store Arc Tagger's weight tables as int8 to shrink arctag.onnx. The C API needs no changes.

  python tools/quantize_model.py model/arctag.onnx out/arctag.onnx

Only the weights are int8: each table has one scale per column and is converted back to float when the model loads, and
inference runs in float. Do not use ONNX Runtime's dynamic quantization or MatMulNBits here. They change earlier scores
as more words arrive, which breaks streaming (tests/test_parity.c fails on both). Needs: numpy, onnx.
"""
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

MIN_VALUES = 4096
OPS = {"MatMul", "Gather", "Identity"}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    model = onnx.load(src)
    tables = {t.name: t for t in model.graph.initializer}
    users = {}
    for node in model.graph.node:
        for name in node.input:
            if name in tables:
                users.setdefault(name, set()).add(node.op_type)
    nodes = []
    for name, table in tables.items():
        if table.data_type != onnx.TensorProto.FLOAT or len(table.dims) != 2:
            continue
        if int(np.prod(table.dims)) < MIN_VALUES or not users.get(name, set()) <= OPS:
            continue
        values = numpy_helper.to_array(table)
        scale = np.maximum(np.abs(values).max(axis=0), 1e-8) / 127.0
        model.graph.initializer.remove(table)
        model.graph.initializer.extend([
            numpy_helper.from_array(np.round(values / scale).astype(np.int8), name + "_q"),
            numpy_helper.from_array(scale.astype(np.float32), name + "_s"),
            numpy_helper.from_array(np.zeros(values.shape[1], dtype=np.int8), name + "_z"),
        ])
        nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [name], axis=1))
    for node in reversed(nodes):
        model.graph.node.insert(0, node)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    onnx.save(model, dst)
    print(f"{len(nodes)} tables: {os.path.getsize(src) / 1e6:.1f} MB -> {os.path.getsize(dst) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
