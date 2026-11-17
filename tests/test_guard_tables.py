#!/usr/bin/env python3
"""The Python guard mirror (python/arctag_guard.py) holds the same word lists, patterns and kids thresholds as
src/guard.c and src/kids_thr.inc.   python tests/test_guard_tables.py"""
import os
import re
import sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "python"))
import arctag_guard as G  # noqa: E402

c = open(os.path.join(R, "src", "guard.c")).read()
bad = 0
tabs = re.findall(r'static const char \*([A-Z_]+)\[\]\s*=\s*\{(.*?)NULL\};', c, re.S)
for name, body in tabs:
    want = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
    if getattr(G, name, None) != want:
        print("table differs:", name); bad += 1
inc = dict((k, float(v)) for k, v in re.findall(r'\{"([^"]+)", ([0-9.]+)\}', open(os.path.join(R, "src", "kids_thr.inc")).read()))
if inc != G.KIDS_THR:
    print("KIDS_THR differs"); bad += 1
print(f"{len(tabs)} tables, {len(inc)} thresholds: {'FAIL' if bad else 'ok'}")
sys.exit(1 if bad else 0)
