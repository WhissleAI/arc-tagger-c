/* Native forward pass == ONNX Runtime, per tag, on the same model inputs (tools/make_refs.py). Also the state machinery on the
 * same data: a word's row never depends on later words, one-word steps match batched steps, and a rewind reproduces the rows.
 * usage: test_native_parity WEIGHTS.atw REFS.bin [TOL_PROB TOL_LOGIT [WINDOW]]
 * WINDOW also reports, without judging, how far an attention window of that many words is from the graph's full context on the rows
 * spans are decoded from (the newest 96), for the cases longer than the window. */
#include "atn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *f;
static void *rd(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p || fread(p, 1, n, f) != n) { fprintf(stderr, "refs file: short read\n"); exit(2); }
    return p;
}
int main(int c, char **v) {
    if (c < 3) { fprintf(stderr, "usage: %s WEIGHTS.atw REFS.bin [TOL_PROB TOL_LOGIT]\n", v[0]); return 2; }
    double tol_p = c > 3 ? atof(v[3]) : 2e-4, tol_l = c > 4 ? atof(v[4]) : 5e-4;
    char err[512]; atn_model *m = atn_open(v[1], err, sizeof err);
    if (!m) { fprintf(stderr, "%s\n", err); return 1; }
    if (!(f = fopen(v[2], "rb"))) { fprintf(stderr, "cannot open %s\n", v[2]); return 2; }
    int nc, no, dims[ATN_NOUT], bad = 0;
    if (fread(&nc, 4, 1, f) != 1 || fread(&no, 4, 1, f) != 1 || no != ATN_NOUT || fread(dims, 4, ATN_NOUT, f) != ATN_NOUT) { fprintf(stderr, "refs file: bad header\n"); return 2; }
    for (int i = 0; i < ATN_NOUT; i++) if (dims[i] != atn_out_dim(m, i)) { fprintf(stderr, "output %s: ONNX has %d columns, weights have %d\n", atn_out_name(i), dims[i], atn_out_dim(m, i)); bad = 1; }
    if (bad) return 1;
    int tot = atn_out_total(m), cap = 4096;
    double worst[ATN_NOUT]; int wcase[ATN_NOUT], wrow[ATN_NOUT], wcol[ATN_NOUT]; memset(worst, 0, sizeof worst); memset(wcase, 0, sizeof wcase); memset(wrow, 0, sizeof wrow); memset(wcol, 0, sizeof wcol);
    double gmax = 0, lmax = 0, sum = 0, rev_max = 0, step_max = 0; long cnt = 0; int nans = 0, words = 0, rev_cases = 0;
    atn *a = atn_new(m, 0, cap, cap);
    if (!a) { fprintf(stderr, "out of memory\n"); return 1; }
    int window = c > 5 ? atoi(v[5]) : 0; atn *aw = window ? atn_new(m, 0, window + ATN_REWIND_WORDS, window) : NULL; double win_max = 0, win_sum = 0; long win_cnt = 0; int win_w = 0; const char *win_at = "";
    for (int k = 0; k < nc; k++) {
        int W, dom;
        if (fread(&W, 4, 1, f) != 1 || fread(&dom, 4, 1, f) != 1 || W < 1 || W > cap) { fprintf(stderr, "refs file: bad case\n"); return 2; }
        size_t Z = (size_t)W;
        int *pcs = rd(4 * Z * 8), *npc = rd(4 * Z), *wfl = rd(4 * Z * 5), *feats = rd(4 * Z * 8), *spk = rd(4 * Z), *ust = rd(4 * Z);
        atn_word *ws = calloc(Z, sizeof *ws); float *o = malloc(sizeof(float) * Z * tot), *o2 = malloc(sizeof(float) * tot);
        if (!ws || !o || !o2) { fprintf(stderr, "out of memory\n"); return 1; }
        for (int i = 0; i < W; i++) { memcpy(ws[i].pcs, pcs + i * 8, 32); ws[i].npc = npc[i]; memcpy(ws[i].wfl, wfl + i * 5, 20); memcpy(ws[i].feats, feats + i * 8, 32); ws[i].spk = spk[i]; ws[i].ust = ust[i]; }
        if (atn_set_domain(a, dom)) { fprintf(stderr, "domain %d out of range\n", dom); return 1; }
        atn_steps(a, ws, W, o);
        words += W;
        float *ow = NULL;
        if (aw && W > window) { ow = malloc(sizeof(float) * Z * tot); atn_set_domain(aw, dom); atn_steps(aw, ws, W, ow); win_w = W; }
        for (int oi = 0, off = 0; oi < ATN_NOUT; off += dims[oi], oi++) {
            float *r = rd(sizeof(float) * Z * dims[oi]);
            for (int i = 0; i < W; i++) for (int j = 0; j < dims[oi]; j++) {
                double d = fabs((double)r[i * dims[oi] + j] - (double)o[(size_t)i * tot + off + j]);
                if (isnan(d)) { nans++; continue; }
                if (oi < ATN_NOUT - 2) { sum += d; cnt++; if (d > gmax) gmax = d; } else if (d > lmax) lmax = d;
                if (d > worst[oi]) { worst[oi] = d; wcase[oi] = k; wrow[oi] = i; wcol[oi] = j; }
                if (ow && oi < ATN_NOUT - 2 && i >= W - 96) { double e = fabs((double)r[i * dims[oi] + j] - (double)ow[(size_t)i * tot + off + j]); win_sum += e; win_cnt++; if (e > win_max) { win_max = e; win_at = atn_out_name(oi); } }
            }
            free(r);
        }
        free(ow);
        /* a rewind and replay must reproduce the rows (rounding aside: replays may batch differently) */
        if (W >= 40) {
            int p = atn_rewind(a, W - 7);
            if (p < 0) { fprintf(stderr, "case %d: no snapshot within 7 words\n", k); bad = 1; }
            else {
                rev_cases++;
                for (int i = p; i < W; i++) {
                    atn_step(a, ws + i, o2);
                    for (int q = 0; q < tot; q++) { double d = fabs((double)o2[q] - (double)o[(size_t)i * tot + q]); if (d > rev_max) rev_max = d; }
                }
            }
        }
        /* one word at a time == batched, on the short cases */
        if (W <= 40) {
            atn_reset(a);
            for (int i = 0; i < W; i++) {
                atn_step(a, ws + i, o2);
                for (int q = 0; q < tot; q++) { double d = fabs((double)o2[q] - (double)o[(size_t)i * tot + q]); if (d > step_max) step_max = d; }
            }
        }
        free(pcs); free(npc); free(wfl); free(feats); free(spk); free(ust); free(ws); free(o); free(o2);
    }
    for (int oi = 0; oi < ATN_NOUT; oi++) printf("  %-24s max %.3e at case %d word %d col %d\n", atn_out_name(oi), worst[oi], wcase[oi], wrow[oi], wcol[oi]);
    printf("parity (%s weights, %s kernels, %s matmul): %d cases, %d words, %ld values: probabilities max %.3e mean %.3e (tol %.1e), logits max %.3e (tol %.1e), NaN %d\n",
           atn_weights_kind(m), atn_simd_name(), atn_gemm_name(), nc, words, cnt, gmax, sum / (double)(cnt ? cnt : 1), tol_p, lmax, tol_l, nans);
    printf("state: rewind + replay max diff %.3e over %d cases; one-word steps vs batched max diff %.3e\n", rev_max, rev_cases, step_max);
    if (aw) { printf("window %d vs full context, newest 96 rows, longest case %d words: max %.3f (in %s), mean %.2e, over %ld values\n", window, win_w, win_max, win_at, win_sum / (double)(win_cnt ? win_cnt : 1), win_cnt); atn_free(aw); }
    if (gmax > tol_p) { fprintf(stderr, "probabilities differ by %.3e, more than %.1e\n", gmax, tol_p); bad = 1; }
    if (lmax > tol_l) { fprintf(stderr, "logits differ by %.3e, more than %.1e\n", lmax, tol_l); bad = 1; }
    if (nans) { fprintf(stderr, "%d NaN differences\n", nans); bad = 1; }
    if (rev_max > tol_l || step_max > tol_l) { fprintf(stderr, "state check: rewind or single-step rows differ by more than %.1e\n", tol_l); bad = 1; }
    atn_free(a); atn_close(m); fclose(f);
    return bad ? 1 : 0;
}
