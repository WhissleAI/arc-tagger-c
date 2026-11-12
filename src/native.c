/* See native.h. Layout conventions: activations are float [D]; a table W [K, N] multiplies x from the left (y = x W). */
#include "native.h"
#include "onnx_file.h"
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ build configuration
 * No flags are needed. Apple: Accelerate does the batched products (link -framework Accelerate), NAT_NO_ACCELERATE opts out.
 * arm64: NEON. x86_64: AVX2 + FMA, enabled for the kernels below by a pragma, so a universal build needs no per-arch flags;
 * the CPU is checked at load. Anything else, or NAT_SCALAR: plain C. */
#if defined(__APPLE__) && !defined(NAT_NO_ACCELERATE)
#define NAT_ACCELERATE 1
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <pthread.h>
#endif
#if !defined(NAT_SCALAR) && defined(__aarch64__)
#define NAT_NEON 1
#include <arm_neon.h>
#elif !defined(NAT_SCALAR) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define NAT_AVX2 1
#include <immintrin.h>
#else
#undef NAT_SCALAR
#define NAT_SCALAR 1
#endif

#define D 256
#define NH 4
#define HD 64
#define NS 64
#define NPROJ 1160
#define LWIN 32
#define LRING 128
#define SNAP_EVERY 16
#define SNAP_KEEP 4
#define NBLK 6
#define NLOC 2
#define NFL 5
#define NFEATS 8
#define NSH 10
#define NTD 16
#define MAX_COLS 512
#define MAX_GROUPS 8
#define NEG (-10000.f)
#define BLOCK 32          /* words per batched pass */
#define PANEL_NB 128      /* columns per panel */
#define PANEL_K 512       /* largest K of any table */

enum { DT_F32, DT_I8, DT_I32, DT_U8 };
enum { F = 1 << DT_F32, I8 = 1 << DT_I8, I32 = 1 << DT_I32, U8 = 1 << DT_U8, TBL = F | I8 };

typedef struct { int dt, nd, d[4]; const void *p; const float *s; } tens;  /* DT_I8: s = one scale per column */

enum { Q_FRAME, Q_BEG, Q_IN, Q_END, Q_AT, Q_IMM, Q_REC, Q_UTT, Q_SPK, Q_ITEM, NQ };
static const char *const QN[NQ] = {"frame", "span_beg", "span_in", "span_end", "ev_at", "ev_imm", "ev_rec", "utt", "speaker", "item"};
static const char *const ON[NAT_NOUT] = {"frame", "span_beg", "span_in", "span_end", "ev_at", "ev_imm", "ev_rec", "utt", "speaker", "item",
    "sh_entity_number_t", "sh_entity_number_b", "sh_entity_mention_t", "sh_entity_mention_b", "sh_safety_pii_t", "sh_safety_pii_b",
    "sh_safety_profanity_t", "sh_safety_profanity_b", "sh_safety_toxicity_t", "sh_safety_toxicity_b", "ver_m", "ver_e"};

typedef struct { tens M; const float *b; const int *c, *gi, *gp; int n, ngi, ng; const int *gg[MAX_GROUPS]; int ngg[MAX_GROUPS]; } readout;
typedef struct { tens g, b, Wf, Wout, M1, M2; const float *bf, *nw, *a, *uw, *vw, *ub, *vb, *bout, *mnw, *m1, *m2; } layer_w;
typedef struct { const float *n1w, *n1b, *bqkv, *bo, *n2w, *n2b, *f1, *f2; tens Wqkv, Wo, F1, F2; } mixa_w;
typedef struct { const float *nw, *qnw, *knw, *kmW, *kmb, *rope; float slope[NH], kb[NH]; tens Wqkv, Wo; } mixb_w;

struct nat_model {
    void **blocks; int nblocks, cblocks; size_t bytes;
    char kind[16];
    tens tokE, W1, W2, src, fl[NFL], feat[NFEATS], binp, Winp, tD, tU, valid, shW, rq, rk, rv, ro, rou, sbM;
    tens P1[3], P2[3];
    const float *ppos, *b1, *lnw, *lnb, *locnw, *locnb, *in_nw, *in_nb, *onw, *q0, *rnw, *rqnw, *rknw, *rkmW, *rkmb,
                *sbrb, *sbk0, *sblnw, *sblnb, *shb, *p1[3], *p2[3], *uaw, *uab, *vw1, *vw2, *vb;
    float rope_a[HD / 2], rope_b[HD / 2], rkb;
    int ndom, shd[NSH];
    mixa_w mixa[NLOC]; layer_w L[NBLK]; mixb_w G[2];
    readout q[NQ];
    int odim[NAT_NOUT], ooff[NAT_NOUT], total;
};

/* ------------------------------------------------------------------ loader */
typedef struct { const ox_graph *g; nat_model *m; char *err; size_t errlen; int failed, kept_i8, folded; } ldctx;

static void ld_err(ldctx *c, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;
static void ld_err(ldctx *c, const char *fmt, ...) {
    if (c->failed++) return;                     /* keep the first problem */
    if (!c->err || !c->errlen) return;
    va_list ap; va_start(ap, fmt); vsnprintf(c->err, c->errlen, fmt, ap); va_end(ap);
}
static void set_msg(char *err, size_t errlen, const char *msg) { if (err && errlen) snprintf(err, errlen, "%s", msg); }

static void *keep(ldctx *c, size_t n) {
    nat_model *m = c->m;
    if (m->nblocks == m->cblocks) {
        int nc = m->cblocks ? 2 * m->cblocks : 512;
        void **b = (void **)realloc(m->blocks, sizeof(void *) * (size_t)nc);
        if (!b) { ld_err(c, "out of memory"); return NULL; }
        m->blocks = b; m->cblocks = nc;
    }
    void *p = malloc(n ? n : 1);
    if (!p) { ld_err(c, "out of memory"); return NULL; }
    m->blocks[m->nblocks++] = p; m->bytes += n;
    return p;
}

static int is_float_t(int dt) { return dt == OX_FLOAT || dt == OX_DOUBLE; }

/* a plain tensor: float, int32 (from any integer type, range checked) or uint8 (bool / uint8) */
static int from_plain(ldctx *c, const ox_tensor *t, int dtmask, const char *nm, tens *o) {
    if (t->nd > 4) { ld_err(c, "model: table %s has rank %d", nm, t->nd); return -1; }
    o->nd = t->nd; for (int k = 0; k < t->nd; k++) { if (t->d[k] > INT32_MAX) { ld_err(c, "model: table %s is too large", nm); return -1; } o->d[k] = (int)t->d[k]; }
    size_t n = (size_t)t->count;
    if (is_float_t(t->dt)) {
        if (!(dtmask & F)) { ld_err(c, "model: table %s is float, an index table was expected", nm); return -1; }
        float *p = (float *)keep(c, n * 4);
        if (!p || ox_read(t, OX_FLOAT, p, t->count)) { ld_err(c, "model: table %s: bad data", nm); return -1; }
        o->dt = DT_F32; o->p = p;
        return 0;
    }
    if ((dtmask & U8) && (t->dt == OX_BOOL || t->dt == OX_UINT8)) {
        uint8_t *p = (uint8_t *)keep(c, n);
        if (!p || ox_read(t, OX_UINT8, p, t->count)) { ld_err(c, "model: table %s: bad data", nm); return -1; }
        o->dt = DT_U8; o->p = p;
        return 0;
    }
    if (dtmask & I32) {
        int64_t *tmp = (int64_t *)malloc(n * 8 + 8);
        int32_t *p = (int32_t *)keep(c, n * 4);
        if (!tmp || !p || ox_read(t, OX_INT64, tmp, t->count)) { free(tmp); ld_err(c, "model: table %s: bad data", nm); return -1; }
        for (size_t i = 0; i < n; i++) {
            if (tmp[i] < INT32_MIN || tmp[i] > INT32_MAX) { free(tmp); ld_err(c, "model: table %s: value out of range", nm); return -1; }
            p[i] = (int32_t)tmp[i];
        }
        free(tmp);
        o->dt = DT_I32; o->p = p;
        return 0;
    }
    ld_err(c, "model: table %s has element type %d", nm, t->dt);
    return -1;
}

/* An 8-bit table q [K, N] with scale s and zero point z (per tensor, or per column if axis = 1, or per row if axis = 0).
 * int8 with zero points 0 and per-column or per-tensor scales stays int8 (when the caller accepts it); anything else is
 * converted to float. */
static int from_8bit(ldctx *c, const ox_tensor *q, const ox_tensor *s, const ox_tensor *z, int64_t axis, int dtmask, const char *nm, tens *o) {
    if (q->nd != 2 || (q->dt != OX_INT8 && q->dt != OX_UINT8) || !s || !is_float_t(s->dt)) { ld_err(c, "model: quantised table %s: unsupported layout", nm); return -1; }
    if (q->d[0] > INT32_MAX || q->d[1] > INT32_MAX) { ld_err(c, "model: table %s is too large", nm); return -1; }
    int K = (int)q->d[0], N = (int)q->d[1];
    int64_t ns = s->count;
    int per = ns == 1 ? 0 : (axis == 1 || axis == -1) && ns == N ? 1 : axis == 0 && ns == K ? 2 : -1;
    if (per < 0 || (z && (z->count != ns || z->dt != q->dt))) { ld_err(c, "model: quantised table %s: scale or zero point shape", nm); return -1; }
    float *sc = (float *)malloc(sizeof(float) * (size_t)ns);
    uint8_t *zp = (uint8_t *)calloc((size_t)ns, 1);
    uint8_t *qv = (uint8_t *)malloc((size_t)K * (size_t)N + 1);
    int rc = -1, zero = 1;
    if (!sc || !zp || !qv) { ld_err(c, "out of memory"); goto done; }
    if (ox_read(s, OX_FLOAT, sc, ns) || (z && ox_read(z, q->dt, zp, ns)) || ox_read(q, q->dt, qv, (int64_t)K * N)) { ld_err(c, "model: quantised table %s: bad data", nm); goto done; }
    for (int64_t i = 0; i < ns; i++) if (zp[i]) zero = 0;
    o->nd = 2; o->d[0] = K; o->d[1] = N;
    if (q->dt == OX_INT8 && zero && per != 2 && (dtmask & I8)) {
        int8_t *p = (int8_t *)keep(c, (size_t)K * N); float *ps = (float *)keep(c, sizeof(float) * (size_t)N);
        if (!p || !ps) goto done;
        memcpy(p, qv, (size_t)K * N);
        for (int j = 0; j < N; j++) ps[j] = sc[per == 1 ? j : 0];
        o->dt = DT_I8; o->p = p; o->s = ps; c->kept_i8 = 1;
    } else {
        if (!(dtmask & F)) { ld_err(c, "model: table %s: quantised, an index table was expected", nm); goto done; }
        float *p = (float *)keep(c, sizeof(float) * (size_t)K * N);
        if (!p) goto done;
        for (int i = 0; i < K; i++)
            for (int j = 0; j < N; j++) {
                int k = per == 1 ? j : per == 2 ? i : 0;
                int v = q->dt == OX_INT8 ? (int)(int8_t)qv[(size_t)i * N + j] - (int)(int8_t)zp[k] : (int)qv[(size_t)i * N + j] - (int)zp[k];
                p[(size_t)i * N + j] = (float)v * sc[k];
            }
        o->dt = DT_F32; o->p = p; c->folded = 1;
    }
    rc = 0;
done:
    free(sc); free(zp); free(qv);
    return rc;
}

/* look a table up by name: a plain value (initializer, Constant, Identity), the output of a DequantizeLinear, or a
 * "_quantized" / "_scale" / "_zero_point" triple. Then check its type and shape (-1 = any size). */
static tens need(ldctx *c, int dtmask, int nd, int d0, int d1, int d2, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 7, 8)))
#endif
    ;
static tens need(ldctx *c, int dtmask, int nd, int d0, int d1, int d2, const char *fmt, ...) {
    char b[96], nm[128]; va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    tens t; memset(&t, 0, sizeof t);
    if (c->failed) return t;
    snprintf(nm, sizeof nm, "core.%s", b);
    const ox_tensor *v = ox_value(c->g, nm);
    int rc;
    if (v) rc = from_plain(c, v, dtmask, b, &t);
    else {
        const ox_node *n = ox_producer(c->g, nm);
        if (n && !strcmp(n->op, "DequantizeLinear") && n->nin >= 2) {
            rc = from_8bit(c, ox_value(c->g, n->in[0]) ? ox_value(c->g, n->in[0]) : NULL, ox_value(c->g, n->in[1]),
                           n->nin > 2 ? ox_value(c->g, n->in[2]) : NULL, n->has_axis ? n->axis : 1, dtmask, b, &t);
            if (!ox_value(c->g, n->in[0]) && !c->failed) { ld_err(c, "model: table %s: computed quantised input", b); rc = -1; }
        } else {
            char qn[160], sn[160], zn[160];
            snprintf(qn, sizeof qn, "%s_quantized", nm); snprintf(sn, sizeof sn, "%s_scale", nm); snprintf(zn, sizeof zn, "%s_zero_point", nm);
            const ox_tensor *q = ox_init(c->g, qn);
            if (!q) { ld_err(c, "model: table %s is missing (is this an Arc Tagger model?)", b); return t; }
            rc = from_8bit(c, q, ox_init(c->g, sn), ox_init(c->g, zn), 1, dtmask, b, &t);
        }
    }
    if (rc) { memset(&t, 0, sizeof t); return t; }
    int want[3] = {d0, d1, d2}, ok = t.nd == nd && (dtmask >> t.dt & 1);
    for (int k = 0; k < nd && k < 3; k++) if (want[k] >= 0 && t.d[k] != want[k]) ok = 0;
    if (!ok) {
        ld_err(c, "model: table %s has type %d shape [%d %d %d %d] rank %d; this runtime expects rank %d shape [%d %d %d] (-1 = any). "
               "Is this a different model?", b, t.dt, t.d[0], t.d[1], t.d[2], t.d[3], t.nd, nd, d0, d1, d2);
        memset(&t, 0, sizeof t);
    }
    return t;
}
#define FV(n, ...) ((const float *)need(c, F, 1, n, -1, -1, __VA_ARGS__).p)                 /* float vector of size n */
#define TB(k, n, ...) need(c, TBL, 2, k, n, -1, __VA_ARGS__)                                 /* table [k, n]: f32 or int8 */

/* ---- values the graph holds without a stable name, found from where they are used */
static const ox_tensor *const_in(const ox_graph *g, const ox_node *n, int64_t count) {
    const ox_tensor *hit = NULL;
    if (!n) return NULL;
    for (int k = 0; k < n->nin; k++) {
        const ox_tensor *v = ox_value(g, n->in[k]);
        if (v && is_float_t(v->dt) && v->count == count) { if (hit) return NULL; hit = v; }
    }
    return hit;
}
static const ox_node *chain(const ox_graph *g, const char *from, const char *const *ops, int nops) {
    const ox_node *n = NULL; const char *x = from;
    for (int i = 0; i < nops; i++) {
        n = ox_only_consumer(g, x, ops[i]);
        if (!n || n->nout < 1) return NULL;
        x = n->out[0];
    }
    return n;
}
static void find_constants(ldctx *c) {
    const ox_graph *g = c->g; nat_model *m = c->m;
    for (int k = 0; k < 2 && !c->failed; k++) {
        char nm[64]; snprintf(nm, sizeof nm, "core.G%d_kmb", k ? 3 : 1);
        static const char *const P[] = {"Add", "Sigmoid", "Unsqueeze", "Mul"};
        const ox_node *mul = chain(g, nm, P, 4);
        const ox_tensor *kb = const_in(g, mul, NH);
        const ox_node *add = mul ? ox_only_consumer(g, mul->out[0], "Add") : NULL;
        const ox_node *other = NULL;
        for (int i = 0; add && i < add->nin; i++) if (strcmp(add->in[i], mul->out[0])) other = ox_producer(g, add->in[i]);
        const ox_tensor *sl = other && !strcmp(other->op, "Mul") ? const_in(g, other, NH) : NULL;
        if (!kb || !sl || ox_read(kb, OX_FLOAT, m->G[k].kb, NH) || ox_read(sl, OX_FLOAT, m->G[k].slope, NH)) { ld_err(c, "model: graph layout not recognised near %s", nm + 5); return; }
    }
    int found = 0;
    for (int i = 0; i < g->nnode && !c->failed; i++) {
        const ox_node *n = &g->node[i];
        if (strcmp(n->op, "Mul")) continue;
        for (int k = 0; k < n->nin; k++) {
            const ox_tensor *v = ox_value(g, n->in[k]);
            if (!v || !is_float_t(v->dt) || v->nd != 2 || v->d[0] != 1 || v->d[1] != HD / 2) continue;
            if (found < 2 && !ox_read(v, OX_FLOAT, found ? m->rope_b : m->rope_a, HD / 2)) found++;
            else found = 3;
        }
    }
    if (!c->failed && found != 2) { ld_err(c, "model: graph layout not recognised (position tables)"); return; }
    static const char *const R[] = {"Add", "Sigmoid", "Log", "Mul"};
    const ox_tensor *kb = const_in(g, chain(g, "core.mr_kmb", R, 4), 1);
    if (!c->failed && (!kb || ox_read(kb, OX_FLOAT, &m->rkb, 1))) ld_err(c, "model: graph layout not recognised near mr_kmb");
}

static int check_idx(ldctx *c, const int *p, int n, int limit, const char *what, const char *q) {
    for (int i = 0; i < n; i++) if (p[i] < 0 || p[i] >= limit) { ld_err(c, "model: %s of %s: index %d out of range (%d, limit %d)", what, q, i, p[i], limit); return -1; }
    return 0;
}

static int load_tensors(nat_model *m, ldctx *c) {
    m->tokE = TB(-1, D, "tokE"); m->W1 = TB(D, D, "W1"); m->W2 = TB(D, D, "W2"); m->src = TB(-1, D, "src");
    m->ppos = (const float *)need(c, F, 2, 8, D, -1, "ppos").p; m->b1 = FV(D, "b1"); m->lnw = FV(D, "lnw"); m->lnb = FV(D, "lnb");
    static const char *const FN[NFEATS] = {"gap", "dur", "fin", "conf", "chg", "asr", "spksrc", "turn"};
    for (int i = 0; i < NFL; i++) m->fl[i] = TB(-1, D, "fl%d", i);
    for (int i = 0; i < NFEATS; i++) m->feat[i] = TB(-1, D, "feat_%s", FN[i]);
    for (int l = 0; l < NLOC; l++) {
        mixa_w *e = &m->mixa[l];
        e->n1w = FV(D, "enc%d_n1w", l); e->n1b = FV(D, "enc%d_n1b", l); e->Wqkv = TB(D, 3 * D, "enc%d_Wqkv", l); e->bqkv = FV(3 * D, "enc%d_bqkv", l);
        e->Wo = TB(D, D, "enc%d_Wo", l); e->bo = FV(D, "enc%d_bo", l); e->n2w = FV(D, "enc%d_n2w", l); e->n2b = FV(D, "enc%d_n2b", l);
        e->F1 = TB(D, 2 * D, "enc%d_F1", l); e->f1 = FV(2 * D, "enc%d_f1", l); e->F2 = TB(2 * D, D, "enc%d_F2", l); e->f2 = FV(D, "enc%d_f2", l);
    }
    m->locnw = FV(D, "encnw"); m->locnb = FV(D, "encnb"); m->in_nw = FV(D, "in_nw"); m->in_nb = FV(D, "in_nb");
    m->Winp = TB(D, D, "Winp"); m->binp = TB(-1, D, "binp");
    int nd = m->binp.d[0]; m->ndom = nd;
    if (!c->failed && nd < 1) ld_err(c, "model: no domains");
    for (int l = 0; l < NBLK; l++) {
        layer_w *L = &m->L[l];
        L->g = TB(nd, D, "L%d_g", l); L->b = TB(nd, D, "L%d_b", l); L->Wf = TB(D, NPROJ, "L%d_Wf", l); L->bf = FV(NPROJ, "L%d_bf", l); L->nw = FV(D, "L%d_nw", l);
        L->a = FV(NH, "L%d_A", l); L->uw = FV(NS, "L%d_Bw", l); L->vw = FV(NS, "L%d_Cw", l);
        L->ub = (const float *)need(c, F, 2, NH, NS, -1, "L%d_Bb", l).p; L->vb = (const float *)need(c, F, 2, NH, NS, -1, "L%d_Cb", l).p;
        L->Wout = TB(D, D, "L%d_Wout", l); L->bout = FV(D, "L%d_bout", l); L->mnw = FV(D, "L%d_mnw", l);
        L->M1 = TB(D, 2 * D, "L%d_M1", l); L->m1 = FV(2 * D, "L%d_m1", l); L->M2 = TB(2 * D, D, "L%d_M2", l); L->m2 = FV(D, "L%d_m2", l);
    }
    for (int k = 0; k < 2; k++) {
        int li = k ? 3 : 1; mixb_w *G = &m->G[k];
        G->nw = FV(D, "G%d_nw", li); G->Wqkv = TB(D, 3 * D, "G%d_Wqkv", li); G->qnw = FV(HD, "G%d_qnw", li); G->knw = FV(HD, "G%d_knw", li);
        G->Wo = TB(D, D, "G%d_Wo", li); G->kmW = FV(D, "G%d_kmW", li); G->kmb = FV(1, "G%d_kmb", li);
    }
    m->tD = need(c, F, 3, nd, D, NTD, "adD"); m->tU = need(c, F, 3, nd, NTD, D, "adU"); m->onw = FV(D, "onw"); m->q0 = FV(D, "dq0");
    m->rnw = FV(D, "mr_nw"); m->rq = TB(D, HD, "mr_q"); m->rk = TB(D, HD, "mr_k"); m->rv = TB(D, HD, "mr_v"); m->rqnw = FV(HD, "mr_qnw"); m->rknw = FV(HD, "mr_knw");
    m->rkmW = FV(D, "mr_kmW"); m->rkmb = FV(1, "mr_kmb"); m->ro = TB(HD, D, "mr_o");
    m->valid = need(c, U8, 2, nd, -1, -1, "valid");
    m->sbrb = FV(D, "war_rb"); m->sbk0 = FV(D, "war_k0"); m->sblnw = FV(D, "war_lnw"); m->sblnb = FV(D, "war_lnb"); m->sbM = TB(D, -1, "war_M");
    m->shW = TB(D, -1, "shW"); m->shb = FV(m->shW.d[1], "shb");
    static const char *const PN[3] = {"utt", "speaker", "item"};
    for (int i = 0; i < 3; i++) {
        m->P1[i] = TB(D, D, "P1_%s", PN[i]); m->p1[i] = FV(D, "p1_%s", PN[i]); m->P2[i] = TB(D, D, "P2_%s", PN[i]); m->p2[i] = FV(D, "p2_%s", PN[i]);
    }
    m->uaw = FV(D, "ua_w"); m->uab = FV(1, "ua_b"); m->vw1 = FV(D, "ver_w1"); m->vw2 = FV(D, "ver_w2"); m->vb = FV(1, "ver_b");
    for (int q = 0; q < NQ; q++) {
        readout *h = &m->q[q];
        if (q == Q_BEG) { h->M = m->sbM; h->b = FV(h->M.d[1], "war_b"); }
        else { h->M = TB(D, -1, "M_%s", QN[q]); h->b = FV(h->M.d[1], "b_%s", QN[q]); }
        h->n = h->M.d[1];
        if (!c->failed && (h->n < 1 || h->n > MAX_COLS)) ld_err(c, "model: read-out %s has %d columns (limit %d)", QN[q], h->n, MAX_COLS);
        h->c = (const int *)need(c, I32, 1, h->n, -1, -1, "c_%s", QN[q]).p; h->gp = (const int *)need(c, I32, 1, h->n, -1, -1, "gp_%s", QN[q]).p;
        tens gi = need(c, I32, 1, -1, -1, -1, "gi_%s", QN[q]); h->gi = (const int *)gi.p; h->ngi = gi.d[0];
        if (c->failed) continue;
        for (int k = 0; k < MAX_GROUPS; k++) {
            char nm[64], full[80]; snprintf(nm, sizeof nm, "gg_%s_%d", QN[q], k); snprintf(full, sizeof full, "core.%s", nm);
            if (!ox_value(c->g, full)) break;
            tens g = need(c, I32, 1, -1, -1, -1, "%s", nm); h->gg[k] = (const int *)g.p; h->ngg[k] = g.d[0]; h->ng = k + 1;
        }
    }
    m->rou = TB(HD, m->q[Q_BEG].n, "mr_ou");
    find_constants(c);
    if (c->failed) return -1;
    for (int q = 0; q < NQ; q++) {               /* index tables: every value is used as an array index */
        readout *h = &m->q[q]; int o = h->ngi;
        if (check_idx(c, h->c, h->n, m->valid.d[1], "c", QN[q]) || check_idx(c, h->gi, h->ngi, h->n, "gi", QN[q])) return -1;
        for (int k = 0; k < h->ng; k++) {
            if (h->ngg[k] < 1 || h->ngg[k] >= MAX_COLS) { ld_err(c, "model: group %d of %s has %d columns", k, QN[q], h->ngg[k]); return -1; }
            if (check_idx(c, h->gg[k], h->ngg[k], h->n, "gg", QN[q])) return -1;
            o += h->ngg[k];
        }
        if (o > MAX_COLS) { ld_err(c, "model: read-out %s has %d probabilities (limit %d)", QN[q], o, MAX_COLS); return -1; }
        if (o < 1 || check_idx(c, h->gp, h->n, o, "gp", QN[q])) { ld_err(c, "model: read-out %s: bad probability map", QN[q]); return -1; }
    }
    /* span outputs: consecutive slices of the shW columns, sized as the graph's outputs say */
    int at = 0;
    for (int f = 0; f < NSH; f++) {
        const ox_output *o = NULL;
        for (int i = 0; i < c->g->nout; i++) if (!strcmp(c->g->out[i].name, ON[NQ + f])) o = &c->g->out[i];
        int n = o && o->nd == 2 ? (int)o->d[1] : -1;
        if (n < 1 || n > MAX_COLS) { ld_err(c, "model: output %s is missing or has no fixed width", ON[NQ + f]); return -1; }
        m->shd[f] = n; at += n;
    }
    if (at != m->shW.d[1]) { ld_err(c, "model: span outputs cover %d of %d columns", at, m->shW.d[1]); return -1; }
    for (int i = 0; i < NFL; i++) if (m->fl[i].d[0] < 1) { ld_err(c, "model: empty table fl%d", i); return -1; }
    for (int i = 0; i < NFEATS; i++) if (m->feat[i].d[0] < 1) { ld_err(c, "model: empty table feat_%s", FN[i]); return -1; }
    if (m->tokE.d[0] < 1 || m->src.d[0] < 1) { ld_err(c, "model: empty embedding table"); return -1; }
    int o = 0;
    for (int i = 0; i < NAT_NOUT; i++) {
        int dm = i < NQ ? m->q[i].n : i < NQ + NSH ? m->shd[i - NQ] : 1;
        m->odim[i] = dm; m->ooff[i] = o; o += dm;
    }
    m->total = o;
    snprintf(m->kind, sizeof m->kind, "%s", c->kept_i8 ? "int8" : c->folded ? "int8-folded" : "fp32");
    return 0;
}

static int cpu_ok(char *err, size_t errlen) {
#ifdef NAT_AVX2
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) { set_msg(err, errlen, "this x86_64 CPU does not report AVX2 + FMA, which the native backend needs (use the onnxruntime backend)"); return -1; }
#else
    (void)err; (void)errlen;
#endif
    return 0;
}

void nat_close(nat_model *m) {
    if (!m) return;
    for (int i = 0; i < m->nblocks; i++) free(m->blocks[i]);
    free(m->blocks); free(m);
}

static nat_model *build(const ox_graph *g, char *err, size_t errlen) {
    nat_model *m = (nat_model *)calloc(1, sizeof *m);
    if (!m) { set_msg(err, errlen, "out of memory"); return NULL; }
    ldctx c = {g, m, err, errlen, 0, 0, 0};
    if (load_tensors(m, &c)) { nat_close(m); return NULL; }
    return m;
}

nat_model *nat_open_mem(const void *buf, size_t size, char *err, size_t errlen) {
    if (cpu_ok(err, errlen)) return NULL;
    if (!buf || !size) { set_msg(err, errlen, "model: empty buffer"); return NULL; }
    ox_graph g;
    if (ox_parse(&g, (const uint8_t *)buf, size, err, errlen)) return NULL;
    nat_model *m = build(&g, err, errlen);
    ox_free(&g);
    return m;
}

nat_model *nat_open(const char *path, char *err, size_t errlen) {
    if (cpu_ok(err, errlen)) return NULL;
    ox_graph g;
    if (ox_load(&g, path, err, errlen)) return NULL;
    nat_model *m = build(&g, err, errlen);
    ox_free(&g);
    if (!m && err && errlen) {
        char tmp[512]; snprintf(tmp, sizeof tmp, "%s", err);
        snprintf(err, errlen, "%s: %s", path, tmp);
    }
    return m;
}
int nat_out_dim(const nat_model *m, int o) { return m->odim[o]; }
int nat_out_total(const nat_model *m) { return m->total; }
const char *nat_out_name(int o) { return ON[o]; }
int nat_num_domains(const nat_model *m) { return m->ndom; }
const char *nat_weights_kind(const nat_model *m) { return m->kind; }
size_t nat_weights_bytes(const nat_model *m) { return m->bytes; }
const char *nat_simd_name(void) {
#if defined(NAT_NEON)
    return "neon";
#elif defined(NAT_AVX2)
    return "avx2";
#else
    return "scalar";
#endif
}

/* ------------------------------------------------------------------ Accelerate: one thread, without touching the host's other work */
#ifdef NAT_ACCELERATE
static void legacy_one_thread(void) { setenv("VECLIB_MAXIMUM_THREADS", "1", 0); }
typedef struct { int active; enum BLAS_THREADING prev; } blas_guard;
static blas_guard blas_enter(void) {
    blas_guard g = {0, BLAS_THREADING_MULTI_THREADED};
    if (__builtin_available(macOS 15.0, iOS 18.0, *)) {      /* per thread: restored on the way out */
        g.prev = BLASGetThreading(); g.active = 1;
        BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    } else {                                                  /* older systems: process-wide, and only if the host did not set it */
        static pthread_once_t once = PTHREAD_ONCE_INIT;
        pthread_once(&once, legacy_one_thread);
    }
    return g;
}
static void blas_leave(blas_guard g) {
    if (__builtin_available(macOS 15.0, iOS 18.0, *)) { if (g.active) BLASSetThreading(g.prev); }
}
#else
typedef int blas_guard;
static blas_guard blas_enter(void) { return 0; }
static void blas_leave(blas_guard g) { (void)g; }
#endif

/* ------------------------------------------------------------------ vector layer: VL float lanes */
#ifdef NAT_AVX2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#endif
#define VL 8
typedef __m256 vf;
static inline vf v_ld(const float *p) { return _mm256_loadu_ps(p); }
static inline void v_st(float *p, vf v) { _mm256_storeu_ps(p, v); }
static inline vf v_set(float x) { return _mm256_set1_ps(x); }
static inline vf v_add(vf a, vf b) { return _mm256_add_ps(a, b); }
static inline vf v_sub(vf a, vf b) { return _mm256_sub_ps(a, b); }
static inline vf v_mul(vf a, vf b) { return _mm256_mul_ps(a, b); }
static inline vf v_div(vf a, vf b) { return _mm256_div_ps(a, b); }
static inline vf v_fma(vf a, vf b, vf c) { return _mm256_fmadd_ps(b, c, a); }       /* a + b c */
static inline vf v_fms(vf a, vf b, vf c) { return _mm256_fnmadd_ps(b, c, a); }      /* a - b c */
static inline vf v_min(vf a, vf b) { return _mm256_min_ps(a, b); }
static inline vf v_max(vf a, vf b) { return _mm256_max_ps(a, b); }
static inline vf v_abs(vf a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.f), a); }
static inline vf v_neg(vf a) { return _mm256_xor_ps(a, _mm256_set1_ps(-0.f)); }
static inline vf v_round(vf a) { return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
static inline vf v_pow2(vf n) { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23)); }
static inline vf v_sel_ge(vf a, vf b, vf x, vf y) { return _mm256_blendv_ps(y, x, _mm256_cmp_ps(a, b, _CMP_GE_OQ)); }  /* a >= b ? x : y */
static inline float v_hsum(vf a) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
static inline void v_i8x2(const int8_t *p, vf *lo, vf *hi) {                          /* 2 VL int8 values as floats */
    *lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)p)));
    *hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(p + 8))));
}
#elif defined(NAT_NEON)
#define VL 4
typedef float32x4_t vf;
static inline vf v_ld(const float *p) { return vld1q_f32(p); }
static inline void v_st(float *p, vf v) { vst1q_f32(p, v); }
static inline vf v_set(float x) { return vdupq_n_f32(x); }
static inline vf v_add(vf a, vf b) { return vaddq_f32(a, b); }
static inline vf v_sub(vf a, vf b) { return vsubq_f32(a, b); }
static inline vf v_mul(vf a, vf b) { return vmulq_f32(a, b); }
static inline vf v_div(vf a, vf b) { return vdivq_f32(a, b); }
static inline vf v_fma(vf a, vf b, vf c) { return vfmaq_f32(a, b, c); }
static inline vf v_fms(vf a, vf b, vf c) { return vfmsq_f32(a, b, c); }
static inline vf v_min(vf a, vf b) { return vminq_f32(a, b); }
static inline vf v_max(vf a, vf b) { return vmaxq_f32(a, b); }
static inline vf v_abs(vf a) { return vabsq_f32(a); }
static inline vf v_neg(vf a) { return vnegq_f32(a); }
static inline vf v_round(vf a) { return vrndnq_f32(a); }
static inline vf v_pow2(vf n) { return vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23)); }
static inline vf v_sel_ge(vf a, vf b, vf x, vf y) { return vbslq_f32(vcgeq_f32(a, b), x, y); }
static inline float v_hsum(vf a) { return vaddvq_f32(a); }
static inline void v_i8x2(const int8_t *p, vf *lo, vf *hi) {
    int16x8_t s = vmovl_s8(vld1_s8(p));
    *lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))); *hi = vcvtq_f32_s32(vmovl_high_s16(s));
}
#else
#define VL 1
typedef float vf;
static inline vf v_ld(const float *p) { return *p; }
static inline void v_st(float *p, vf v) { *p = v; }
static inline vf v_set(float x) { return x; }
static inline vf v_add(vf a, vf b) { return a + b; }
static inline vf v_sub(vf a, vf b) { return a - b; }
static inline vf v_mul(vf a, vf b) { return a * b; }
static inline vf v_div(vf a, vf b) { return a / b; }
static inline vf v_fma(vf a, vf b, vf c) { return a + b * c; }
static inline vf v_fms(vf a, vf b, vf c) { return a - b * c; }
static inline vf v_min(vf a, vf b) { return a < b ? a : b; }
static inline vf v_max(vf a, vf b) { return a > b ? a : b; }
static inline vf v_abs(vf a) { return fabsf(a); }
static inline vf v_neg(vf a) { return -a; }
static inline vf v_round(vf a) { return rintf(a); }
static inline vf v_pow2(vf n) { return ldexpf(1.f, (int)n); }
static inline vf v_sel_ge(vf a, vf b, vf x, vf y) { return a >= b ? x : y; }
static inline float v_hsum(vf a) { return a; }
static inline void v_i8x2(const int8_t *p, vf *lo, vf *hi) { *lo = (float)p[0]; *hi = (float)p[1]; }
#endif

/* ------------------------------------------------------------------ kernels */
/* y = x W, W f32 or int8 with per-column scales: rows swept four at a time */
static void mv_f32(const float *x, const float *W, int K, int N, float *y) {
    memset(y, 0, sizeof(float) * (size_t)N);
    int i = 0;
    for (; i + 4 <= K; i += 4) {
        vf x0 = v_set(x[i]), x1 = v_set(x[i + 1]), x2 = v_set(x[i + 2]), x3 = v_set(x[i + 3]);
        const float *w0 = W + (size_t)i * N, *w1 = w0 + N, *w2 = w1 + N, *w3 = w2 + N;
        int j = 0;
        for (; j + 2 * VL <= N; j += 2 * VL) {
            vf a = v_ld(y + j), b = v_ld(y + j + VL);
            a = v_fma(a, x0, v_ld(w0 + j)); b = v_fma(b, x0, v_ld(w0 + j + VL));
            a = v_fma(a, x1, v_ld(w1 + j)); b = v_fma(b, x1, v_ld(w1 + j + VL));
            a = v_fma(a, x2, v_ld(w2 + j)); b = v_fma(b, x2, v_ld(w2 + j + VL));
            a = v_fma(a, x3, v_ld(w3 + j)); b = v_fma(b, x3, v_ld(w3 + j + VL));
            v_st(y + j, a); v_st(y + j + VL, b);
        }
        for (; j < N; j++) y[j] += x[i] * w0[j] + x[i + 1] * w1[j] + x[i + 2] * w2[j] + x[i + 3] * w3[j];
    }
    for (; i < K; i++) { const float *w = W + (size_t)i * N; float xi = x[i]; for (int j = 0; j < N; j++) y[j] += xi * w[j]; }
}
static void mv_i8(const float *x, const int8_t *W, const float *s, int K, int N, float *y) {
    memset(y, 0, sizeof(float) * (size_t)N);
    int i = 0;
    for (; i + 4 <= K; i += 4) {
        vf x0 = v_set(x[i]), x1 = v_set(x[i + 1]), x2 = v_set(x[i + 2]), x3 = v_set(x[i + 3]);
        const int8_t *w0 = W + (size_t)i * N, *w1 = w0 + N, *w2 = w1 + N, *w3 = w2 + N;
        int j = 0;
        for (; j + 2 * VL <= N; j += 2 * VL) {
            vf a = v_ld(y + j), b = v_ld(y + j + VL), l, h;
            v_i8x2(w0 + j, &l, &h); a = v_fma(a, x0, l); b = v_fma(b, x0, h);
            v_i8x2(w1 + j, &l, &h); a = v_fma(a, x1, l); b = v_fma(b, x1, h);
            v_i8x2(w2 + j, &l, &h); a = v_fma(a, x2, l); b = v_fma(b, x2, h);
            v_i8x2(w3 + j, &l, &h); a = v_fma(a, x3, l); b = v_fma(b, x3, h);
            v_st(y + j, a); v_st(y + j + VL, b);
        }
        for (; j < N; j++) y[j] += x[i] * (float)w0[j] + x[i + 1] * (float)w1[j] + x[i + 2] * (float)w2[j] + x[i + 3] * (float)w3[j];
    }
    for (; i < K; i++) { const int8_t *w = W + (size_t)i * N; float xi = x[i]; for (int j = 0; j < N; j++) y[j] += xi * (float)w[j]; }
    int j = 0;
    for (; j + VL <= N; j += VL) v_st(y + j, v_mul(v_ld(y + j), v_ld(s + j)));
    for (; j < N; j++) y[j] *= s[j];
}
static inline void mv(const float *x, const tens *W, float *y) {
    if (W->dt == DT_I8) mv_i8(x, (const int8_t *)W->p, W->s, W->d[0], W->d[1], y);
    else mv_f32(x, (const float *)W->p, W->d[0], W->d[1], y);
}
/* nb columns of an int8 table as floats: the conversion is shared by all the rows of a batched pass */
static void panel_i8(const tens *W, int j0, int nb, float *panel) {
    const int8_t *q = (const int8_t *)W->p; const float *sc = W->s + j0; int K = W->d[0], N = W->d[1];
    for (int i = 0; i < K; i++) {
        const int8_t *r = q + (size_t)i * N + j0; float *o = panel + (size_t)i * nb; int j = 0;
        for (; j + 2 * VL <= nb; j += 2 * VL) { vf l, h; v_i8x2(r + j, &l, &h); v_st(o + j, v_mul(l, v_ld(sc + j))); v_st(o + j + VL, v_mul(h, v_ld(sc + j + VL))); }
        for (; j < nb; j++) o[j] = (float)r[j] * sc[j];
    }
}
#ifndef NAT_ACCELERATE
/* y[T, nb] (stride N) = x[T, K] P[K, nb] (stride ps): four rows at a time, for builds without Accelerate */
static void gemm_panel(const float *x, int T, int K, const float *P, int ps, int nb, float *y, int N) {
    int t = 0;
    for (; t + 4 <= T; t += 4) {
        const float *x0 = x + (size_t)t * K, *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
        float *y0 = y + (size_t)t * N, *y1 = y0 + N, *y2 = y1 + N, *y3 = y2 + N;
        int j = 0;
        for (; j + 2 * VL <= nb; j += 2 * VL) {
            vf z = v_set(0.f), a0 = z, b0 = z, a1 = z, b1 = z, a2 = z, b2 = z, a3 = z, b3 = z;
            const float *p = P + j;
            for (int k = 0; k < K; k++, p += ps) {
                vf pa = v_ld(p), pb = v_ld(p + VL), s;
                s = v_set(x0[k]); a0 = v_fma(a0, s, pa); b0 = v_fma(b0, s, pb);
                s = v_set(x1[k]); a1 = v_fma(a1, s, pa); b1 = v_fma(b1, s, pb);
                s = v_set(x2[k]); a2 = v_fma(a2, s, pa); b2 = v_fma(b2, s, pb);
                s = v_set(x3[k]); a3 = v_fma(a3, s, pa); b3 = v_fma(b3, s, pb);
            }
            v_st(y0 + j, a0); v_st(y0 + j + VL, b0); v_st(y1 + j, a1); v_st(y1 + j + VL, b1);
            v_st(y2 + j, a2); v_st(y2 + j + VL, b2); v_st(y3 + j, a3); v_st(y3 + j + VL, b3);
        }
        for (; j < nb; j++) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0; const float *p = P + j;
            for (int k = 0; k < K; k++, p += ps) { s0 += x0[k] * *p; s1 += x1[k] * *p; s2 += x2[k] * *p; s3 += x3[k] * *p; }
            y0[j] = s0; y1[j] = s1; y2[j] = s2; y3[j] = s3;
        }
    }
    for (; t < T; t++) {
        const float *x0 = x + (size_t)t * K; float *y0 = y + (size_t)t * N;
        for (int j = 0; j < nb; j++) { float s0 = 0; const float *p = P + j; for (int k = 0; k < K; k++, p += ps) s0 += x0[k] * *p; y0[j] = s0; }
    }
}
#endif
/* y[T, N] = x[T, K] W. A few rows: the row sweep (no panel cost). More: a batched pass over panels of columns. */
#ifdef NAT_ACCELERATE
#define MM_BATCH_FROM 3
#else
#define MM_BATCH_FROM 6
#endif
static void mm(float *panel, const float *x, int T, const tens *W, float *y) {
    int K = W->d[0], N = W->d[1];
    if (T < MM_BATCH_FROM) { for (int t = 0; t < T; t++) mv(x + (size_t)t * K, W, y + (size_t)t * N); return; }
#ifdef NAT_ACCELERATE
    if (W->dt != DT_I8) { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, N, K, 1.f, x, K, (const float *)W->p, N, 0.f, y, N); return; }
    for (int j0 = 0; j0 < N; j0 += PANEL_NB) {
        int nb = N - j0 < PANEL_NB ? N - j0 : PANEL_NB;
        panel_i8(W, j0, nb, panel);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, nb, K, 1.f, x, K, panel, nb, 0.f, y + j0, N);
    }
#else
    for (int j0 = 0; j0 < N; j0 += PANEL_NB) {
        int nb = N - j0 < PANEL_NB ? N - j0 : PANEL_NB;
        if (W->dt == DT_I8) { panel_i8(W, j0, nb, panel); gemm_panel(x, T, K, panel, nb, nb, y + j0, N); }
        else gemm_panel(x, T, K, (const float *)W->p + j0, N, nb, y + j0, N);
    }
#endif
}
static inline void addv(float *y, const float *b, int n) { int j = 0; for (; j + VL <= n; j += VL) v_st(y + j, v_add(v_ld(y + j), v_ld(b + j))); for (; j < n; j++) y[j] += b[j]; }
static inline void mulv(float *y, const float *b, int n) { int j = 0; for (; j + VL <= n; j += VL) v_st(y + j, v_mul(v_ld(y + j), v_ld(b + j))); for (; j < n; j++) y[j] *= b[j]; }
static inline float dotv(const float *a, const float *b, int n) {
    vf s = v_set(0.f), t = s, u = s, v = s; int j = 0;
    for (; j + 4 * VL <= n; j += 4 * VL) {
        s = v_fma(s, v_ld(a + j), v_ld(b + j)); t = v_fma(t, v_ld(a + j + VL), v_ld(b + j + VL));
        u = v_fma(u, v_ld(a + j + 2 * VL), v_ld(b + j + 2 * VL)); v = v_fma(v, v_ld(a + j + 3 * VL), v_ld(b + j + 3 * VL));
    }
    float r = v_hsum(v_add(v_add(s, t), v_add(u, v)));
    for (; j < n; j++) r += a[j] * b[j];
    return r;
}
static inline void axpy(float *y, float a, const float *x, int n) {
    vf av = v_set(a); int j = 0;
    for (; j + VL <= n; j += VL) v_st(y + j, v_fma(v_ld(y + j), av, v_ld(x + j)));
    for (; j < n; j++) y[j] += a * x[j];
}
static inline void row_f(const tens *t, int r, float *out) {   /* row of a [R, N] table as floats */
    int N = t->d[1];
    if (t->dt == DT_I8) {
        const int8_t *q = (const int8_t *)t->p + (size_t)r * N; int j = 0;
        for (; j + 2 * VL <= N; j += 2 * VL) {
            vf l, h; v_i8x2(q + j, &l, &h);
            v_st(out + j, v_mul(l, v_ld(t->s + j))); v_st(out + j + VL, v_mul(h, v_ld(t->s + j + VL)));
        }
        for (; j < N; j++) out[j] = (float)q[j] * t->s[j];
    } else memcpy(out, (const float *)t->p + (size_t)r * N, sizeof(float) * (size_t)N);
}
static inline void row_add(const tens *t, int r, float *y) {   /* tables of width D */
    if (t->dt == DT_I8) { float tmp[D]; row_f(t, r, tmp); addv(y, tmp, D); }
    else addv(y, (const float *)t->p + (size_t)r * D, D);
}
/* vector exp (float), about 2 ulp over the useful range; below -87 flushes to 0 */
static inline vf vexp(vf x) {
    const vf inv = v_set(1.44269504088896341f), hi = v_set(0.693359375f), lo = v_set(-2.12194440e-4f), floor_ = v_set(-87.3f);
    vf xc = v_min(v_max(x, floor_), v_set(88.3f));
    vf n = v_round(v_mul(xc, inv));
    vf r = v_fms(v_fms(xc, n, hi), n, lo);
    vf p = v_set(1.9875691500E-4f);
    p = v_fma(v_set(1.3981999507E-3f), p, r);
    p = v_fma(v_set(8.3334519073E-3f), p, r);
    p = v_fma(v_set(4.1665795894E-2f), p, r);
    p = v_fma(v_set(1.6666665459E-1f), p, r);
    p = v_fma(v_set(5.0000001201E-1f), p, r);
    p = v_fma(r, v_mul(p, r), r);
    p = v_add(p, v_set(1.f));
    return v_sel_ge(x, floor_, v_mul(p, v_pow2(n)), v_set(0.f));
}
static inline vf vsigm(vf x) { return v_div(v_set(1.f), v_add(v_set(1.f), vexp(v_neg(x)))); }
/* softmax in place */
static inline void softmax(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    vf m_ = v_set(mx), sum = v_set(0.f); float s = 0; int i = 0;
    for (; i + VL <= n; i += VL) { vf e = vexp(v_sub(v_ld(x + i), m_)); v_st(x + i, e); sum = v_add(sum, e); }
    for (; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    s += v_hsum(sum); float inv = 1.f / s; vf iv = v_set(inv);
    i = 0; for (; i + VL <= n; i += VL) v_st(x + i, v_mul(v_ld(x + i), iv));
    for (; i < n; i++) x[i] *= inv;
}
static inline void lnorm_(float *x, int n, const float *w, const float *b) {   /* eps 1e-5 */
    vf s = v_set(0.f); int j = 0;
    for (; j + VL <= n; j += VL) s = v_add(s, v_ld(x + j));
    float mean = v_hsum(s); for (; j < n; j++) mean += x[j]; mean /= (float)n;
    vf m_ = v_set(mean), v = v_set(0.f); float var; j = 0;
    for (; j + VL <= n; j += VL) { vf d = v_sub(v_ld(x + j), m_); v_st(x + j, d); v = v_fma(v, d, d); }
    var = v_hsum(v); for (; j < n; j++) { x[j] -= mean; var += x[j] * x[j]; }
    float g = 1.f / sqrtf(var / (float)n + 9.999999747378752e-06f);
    vf gv = v_set(g); j = 0;
    for (; j + VL <= n; j += VL) v_st(x + j, v_fma(v_ld(b + j), v_mul(v_ld(x + j), gv), v_ld(w + j)));
    for (; j < n; j++) x[j] = x[j] * g * w[j] + b[j];
}
static inline void rms_(const float *x, int n, const float *w, float *y) {     /* eps = machine eps of float */
    vf v = v_set(0.f); int j = 0; float s;
    for (; j + VL <= n; j += VL) { vf d = v_ld(x + j); v = v_fma(v, d, d); }
    s = v_hsum(v); for (; j < n; j++) s += x[j] * x[j];
    float g = 1.f / sqrtf(s / (float)n + 1.1920928955078125e-07f);
    vf gv = v_set(g); j = 0;
    for (; j + VL <= n; j += VL) v_st(y + j, v_mul(v_mul(v_ld(x + j), gv), v_ld(w + j)));
    for (; j < n; j++) y[j] = x[j] * g * w[j];
}
/* GELU. erfc(x) = t exp(-x^2 + P(t)), t = 1 / (1 + |x| / 2): fractional error about 1.2e-7 (Numerical Recipes erfcc) */
static void gelu(float *x, int n) {
    static const float C[10] = {-1.26551223f, 1.00002368f, 0.37409196f, 0.09678418f, -0.18628806f, 0.27886807f, -1.13520398f, 1.48851587f, -0.82215223f, 0.17087277f};
    const vf rs2 = v_set(0.70710678118654752f), half = v_set(0.5f), one = v_set(1.f), two = v_set(2.f), zero = v_set(0.f);
    int j = 0;
    for (; j + VL <= n; j += VL) {
        vf v = v_ld(x + j), u = v_mul(v, rs2), au = v_abs(u);
        vf t = v_div(one, v_fma(one, half, au));
        vf r = v_set(C[9]);
        for (int k = 8; k >= 0; k--) r = v_fma(v_set(C[k]), r, t);
        vf ec = v_mul(t, vexp(v_fms(r, au, au)));                     /* erfc(|u|) */
        vf ope = v_sel_ge(u, zero, v_sub(two, ec), ec);               /* 1 + erf(u) = u >= 0 ? 2 - ec : ec */
        v_st(x + j, v_mul(v_mul(half, v), ope));
    }
    for (; j < n; j++) x[j] = 0.5f * x[j] * (1.f + erff(x[j] * 0.70710678118654752f));
}


/* ------------------------------------------------------------------ state */
typedef struct { float S[NH][NS][HD], up[NH][NS], xp[NH][HD]; } lay_t;
typedef struct {           /* everything a rewind must restore */
    int n, first;          /* next word, and the word this state began at */
    lay_t lay[NBLK];
    double utt_z[D], utt_w, item_z[D]; double spk_z[10][D]; double spk_n[10];
} rec_t;

typedef struct { float t1[BLOCK * NPROJ], t2[BLOCK * NPROJ], t3[BLOCK * NPROJ], c[BLOCK][HD / 2], s[BLOCK][HD / 2], gc[BLOCK][HD / 2], gs[BLOCK][HD / 2]; } scr_t;

struct nat {
    const nat_model *m; int dom, cap, win;
    float binp[D], dg[NBLK][D], db[NBLK][D]; const float *tD, *tU;   /* the domain's rows */
    unsigned char qvalid[NQ][MAX_COLS];
    rec_t *r;
    rec_t *snap[SNAP_KEEP]; int nsnap;
    float *eK[NLOC], *eV[NLOC];               /* [LRING][NH][HD] */
    float *gK[2], *gV[2], *gM[2];             /* [cap][NH][HD], [cap] */
    float *mK, *mV, *mB;                      /* [cap][HD], [cap] */
    float *score;                             /* [NH][cap + 64] */
    scr_t *scr; float *act[16]; float *panel; int snap_on;
};

static void set_dom(nat *a, int dom) {
    const nat_model *m = a->m; a->dom = dom;
    row_f(&m->binp, dom, a->binp);
    for (int l = 0; l < NBLK; l++) { row_f(&m->L[l].g, dom, a->dg[l]); row_f(&m->L[l].b, dom, a->db[l]); }
    a->tD = (const float *)m->tD.p + (size_t)dom * D * NTD;
    a->tU = (const float *)m->tU.p + (size_t)dom * NTD * D;
    const unsigned char *v = (const unsigned char *)m->valid.p + (size_t)dom * m->valid.d[1];
    for (int q = 0; q < NQ; q++) for (int c = 0; c < m->q[q].n; c++) a->qvalid[q][c] = v[m->q[q].c[c]];
}
nat *nat_new(nat_model *m, int dom, int cap, int win) {
    if (!m || dom < 0 || dom >= m->ndom || cap < 1 || win < 1) return NULL;
    nat *a = (nat *)calloc(1, sizeof *a);
    if (!a) return NULL;
    a->m = m; a->cap = cap; a->win = win < cap ? win : cap;
    int ok = (a->r = (rec_t *)calloc(1, sizeof(rec_t))) != NULL;
    for (int l = 0; l < NLOC; l++) {
        ok &= (a->eK[l] = (float *)malloc(sizeof(float) * LRING * NH * HD)) != NULL;
        ok &= (a->eV[l] = (float *)malloc(sizeof(float) * LRING * NH * HD)) != NULL;
    }
    for (int k = 0; k < 2; k++) {             /* calloc of a large block: pages are committed on first touch */
        ok &= (a->gK[k] = (float *)calloc((size_t)cap * NH * HD, sizeof(float))) != NULL;
        ok &= (a->gV[k] = (float *)calloc((size_t)cap * NH * HD, sizeof(float))) != NULL;
        ok &= (a->gM[k] = (float *)calloc((size_t)cap, sizeof(float))) != NULL;
    }
    ok &= (a->mK = (float *)calloc((size_t)cap * HD, sizeof(float))) != NULL;
    ok &= (a->mV = (float *)calloc((size_t)cap * HD, sizeof(float))) != NULL;
    ok &= (a->mB = (float *)calloc((size_t)cap, sizeof(float))) != NULL;
    ok &= (a->score = (float *)calloc((size_t)NH * (size_t)(cap + 64), sizeof(float))) != NULL;
    ok &= (a->scr = (scr_t *)malloc(sizeof(scr_t))) != NULL;
    ok &= (a->panel = (float *)malloc(sizeof(float) * PANEL_K * PANEL_NB)) != NULL;
    for (int k = 0; k < 16; k++) ok &= (a->act[k] = (float *)malloc(sizeof(float) * BLOCK * D)) != NULL;
    if (!ok) { nat_free(a); return NULL; }
    a->snap_on = 1;
    set_dom(a, dom);
    return a;
}
void nat_free(nat *a) {
    if (!a) return;
    free(a->r); for (int i = 0; i < SNAP_KEEP; i++) free(a->snap[i]);
    for (int l = 0; l < NLOC; l++) { free(a->eK[l]); free(a->eV[l]); }
    for (int k = 0; k < 2; k++) { free(a->gK[k]); free(a->gV[k]); free(a->gM[k]); }
    for (int k = 0; k < 16; k++) free(a->act[k]);
    free(a->scr); free(a->panel); free(a->mK); free(a->mV); free(a->mB); free(a->score); free(a);
}
void nat_restart(nat *a, int pos) { memset(a->r, 0, sizeof(rec_t)); a->r->n = pos; a->r->first = pos; a->nsnap = 0; }
void nat_reset(nat *a) { nat_restart(a, 0); }
int nat_set_domain(nat *a, int dom) {
    if (dom < 0 || dom >= a->m->ndom) return -1;
    set_dom(a, dom); nat_reset(a);
    return 0;
}
int nat_pos(const nat *a) { return a->r->n; }
int nat_first(const nat *a) { return a->r->first; }
int nat_rewind(nat *a, int k) {
    if (k >= a->r->n) return a->r->n;
    int best = -1;
    for (int i = 0; i < a->nsnap; i++) if (a->snap[i]->n <= k && (best < 0 || a->snap[i]->n > a->snap[best]->n)) best = i;
    if (best < 0) return -1;
    memcpy(a->r, a->snap[best], sizeof(rec_t));
    /* snapshots newer than the restored point are stale */
    int o = 0; for (int i = 0; i < a->nsnap; i++) if (a->snap[i]->n <= a->r->n) { rec_t *t = a->snap[o]; a->snap[o] = a->snap[i]; a->snap[i] = t; o++; }
    a->nsnap = o;
    return a->r->n;
}
static void maybe_snap(nat *a) {
    if (!a->snap_on || a->r->n == a->r->first || a->r->n % SNAP_EVERY) return;
    int w;
    if (a->nsnap < SNAP_KEEP) w = a->nsnap;
    else { w = 0; for (int i = 1; i < a->nsnap; i++) if (a->snap[i]->n < a->snap[w]->n) w = i; }
    if (!a->snap[w] && !(a->snap[w] = (rec_t *)malloc(sizeof(rec_t)))) return;   /* out of memory: fewer revision points, nothing else */
    if (w == a->nsnap) a->nsnap++;
    memcpy(a->snap[w], a->r, sizeof(rec_t));
}

/* ------------------------------------------------------------------ the forward pass (T words at a time) */
static inline void rot1(const float *x, const float *c, const float *s, float *y) {
    for (int i = 0; i < HD / 2; i++) { float a = x[2 * i], b = x[2 * i + 1]; y[2 * i] = a * c[i] - b * s[i]; y[2 * i + 1] = a * s[i] + b * c[i]; }
}
static inline void rot_tab(const float *inv, int pos, float *c, float *s) {
    if (pos < 4096) {
        float fp = (float)pos;
        for (int i = 0; i < HD / 2; i++) { float a = fp * inv[i]; c[i] = cosf(a); s[i] = sinf(a); }
    } else {                               /* long streams: keep the angle exact */
        for (int i = 0; i < HD / 2; i++) { float a = (float)fmod((double)pos * (double)inv[i], 6.283185307179586); c[i] = cosf(a); s[i] = sinf(a); }
    }
}

static void mixer_a(nat *a, scr_t *S, int l, int T, float *x) {
    const mixa_w *e = &a->m->mixa[l]; float *pn = a->panel;
    float *t = S->t1, *y = S->t2, *z = S->t3, *KV = a->eK[l], *VV = a->eV[l];
    int i0 = a->r->n, first = a->r->first;
    memcpy(t, x, sizeof(float) * (size_t)T * D);
    for (int w = 0; w < T; w++) lnorm_(t + w * D, D, e->n1w, e->n1b);
    mm(pn, t, T, &e->Wqkv, y);
    float q[BLOCK][NH][HD];
    for (int w = 0; w < T; w++) {
        float *yw = y + w * 3 * D; addv(yw, e->bqkv, 3 * D);
        int i = i0 + w;
        float *kr = KV + (size_t)(i % LRING) * NH * HD, *vr = VV + (size_t)(i % LRING) * NH * HD;
        for (int h = 0; h < NH; h++) {
            rot1(yw + h * HD, S->c[w], S->s[w], q[w][h]);
            rot1(yw + D + h * HD, S->c[w], S->s[w], kr + h * HD);
            memcpy(vr + h * HD, yw + 2 * D + h * HD, sizeof(float) * HD);
        }
    }
    float *out = t;
    for (int w = 0; w < T; w++) {
        int i = i0 + w, lo = i - LWIN + 1; if (lo < first) lo = first;
        int n = i - lo + 1;
        float sc[NH][LWIN], *o = out + w * D; memset(o, 0, sizeof(float) * D);
        for (int j = 0; j < n; j++) {
            const float *kj = KV + (size_t)((lo + j) % LRING) * NH * HD;
            for (int h = 0; h < NH; h++) sc[h][j] = dotv(q[w][h], kj + h * HD, HD) / 8.f;
        }
        for (int h = 0; h < NH; h++) softmax(sc[h], n);
        for (int j = 0; j < n; j++) {
            const float *vj = VV + (size_t)((lo + j) % LRING) * NH * HD;
            for (int h = 0; h < NH; h++) axpy(o + h * HD, sc[h][j], vj + h * HD, HD);
        }
    }
    mm(pn, out, T, &e->Wo, y);
    for (int w = 0; w < T; w++) { addv(x + w * D, y + w * D, D); addv(x + w * D, e->bo, D); }
    memcpy(t, x, sizeof(float) * (size_t)T * D);
    for (int w = 0; w < T; w++) lnorm_(t + w * D, D, e->n2w, e->n2b);
    mm(pn, t, T, &e->F1, z);
    for (int w = 0; w < T; w++) addv(z + w * 2 * D, e->f1, 2 * D);
    gelu(z, T * 2 * D);
    mm(pn, z, T, &e->F2, t);
    for (int w = 0; w < T; w++) { addv(x + w * D, t + w * D, D); addv(x + w * D, e->f2, D); }
}

static inline void softplus4(const float *x, float *y) { for (int h = 0; h < NH; h++) y[h] = log1pf(expf(x[h])); }

static void layer(nat *a, scr_t *S, int l, int T, float *h) {
    const layer_w *L = &a->m->L[l]; float *pn = a->panel;
    lay_t *st = &a->r->lay[l];
    float *nu = S->t1, *pr = S->t2, *ys = S->t3;
    for (int w = 0; w < T; w++) { float *hw = h + w * D; mulv(hw, a->dg[l], D); addv(hw, a->db[l], D); rms_(hw, D, L->nw, nu + w * D); }
    mm(pn, nu, T, &L->Wf, pr);
    for (int w = 0; w < T; w++) {
        float *prw = pr + w * NPROJ; addv(prw, L->bf, NPROJ);
        const float *zi = prw, *xx = prw + D, *up = prw + 2 * D, *vp = prw + 3 * D, *ang = prw + 4 * D, *sp = prw + 4 * D + 128, *mp = sp + 4;
        float *y = ys + w * D; float st_[NH], dec[NH], mix[NH];
        softplus4(sp, st_);
        for (int k = 0; k < NH; k++) { mix[k] = 1.f / (1.f + expf(-mp[k])); dec[k] = expf(st_[k] * L->a[k]); }
        for (int k = 0; k < NH; k++) {
            float u[NS], v[NS];
            rms_(up + k * NS, NS, L->uw, u); addv(u, L->ub + k * NS, NS);
            rms_(vp + k * NS, NS, L->vw, v); addv(v, L->vb + k * NS, NS);
            const float *x = xx + k * HD;
            float g0 = mix[k] * st_[k], g1 = (1.f - mix[k]) * st_[k] * dec[k], al = dec[k];
            float *yh = y + k * HD; memset(yh, 0, sizeof(float) * HD);
            vf alv = v_set(al);
            for (int n = 0; n < NS / 2; n++) {
                float an = ang[k * 32 + n] * st_[k], cs = cosf(an), sn = sinf(an);
                float *sa = st->S[k][2 * n], *sb = st->S[k][2 * n + 1];
                vf bav = v_set(g1 * st->up[k][2 * n]), bbv = v_set(g1 * st->up[k][2 * n + 1]), csv = v_set(cs), snv = v_set(sn);
                vf gav = v_set(g0 * u[2 * n]), gbv = v_set(g0 * u[2 * n + 1]), cav = v_set(v[2 * n]), cbv = v_set(v[2 * n + 1]);
                for (int p = 0; p < HD; p += VL) {
                    vf xpv = v_ld(st->xp[k] + p), xv = v_ld(x + p);
                    vf pa = v_fma(v_mul(alv, v_ld(sa + p)), bav, xpv);
                    vf pb = v_fma(v_mul(alv, v_ld(sb + p)), bbv, xpv);
                    vf na = v_fma(v_fms(v_mul(csv, pa), snv, pb), gav, xv);
                    vf nb = v_fma(v_fma(v_mul(snv, pa), csv, pb), gbv, xv);
                    v_st(sa + p, na); v_st(sb + p, nb);
                    v_st(yh + p, v_fma(v_fma(v_ld(yh + p), cav, na), cbv, nb));
                }
            }
            memcpy(st->up[k], u, sizeof u); memcpy(st->xp[k], x, sizeof(float) * HD);
        }
        for (int j = 0; j + VL <= D; j += VL) { vf z = v_ld(zi + j); v_st(y + j, v_mul(v_ld(y + j), v_mul(z, vsigm(z)))); }
    }
    mm(pn, ys, T, &L->Wout, nu);
    for (int w = 0; w < T; w++) { float *hw = h + w * D; addv(hw, nu + w * D, D); addv(hw, L->bout, D); rms_(hw, D, L->mnw, ys + w * D); }
    mm(pn, ys, T, &L->M1, pr);
    for (int w = 0; w < T; w++) addv(pr + w * 2 * D, L->m1, 2 * D);
    gelu(pr, T * 2 * D);
    mm(pn, pr, T, &L->M2, nu);
    for (int w = 0; w < T; w++) { float *hw = h + w * D; addv(hw, nu + w * D, D); addv(hw, L->m2, D); }
}

static void mixer_b(nat *a, scr_t *S, int k, int T, float *h) {
    const mixb_w *G = &a->m->G[k]; float *pn = a->panel;
    int i0 = a->r->n, first = a->r->first, cap = a->cap, stride = cap + 64;
    float *hn = S->t1, *y = S->t2, *out = S->t3;
    for (int w = 0; w < T; w++) rms_(h + w * D, D, G->nw, hn + w * D);
    mm(pn, hn, T, &G->Wqkv, y);
    float q[BLOCK][NH][HD];
    for (int w = 0; w < T; w++) {
        int i = i0 + w; float *yw = y + w * 3 * D;
        float *Kr = a->gK[k] + (size_t)(i % cap) * NH * HD, *Vr = a->gV[k] + (size_t)(i % cap) * NH * HD;
        for (int b = 0; b < NH; b++) {
            float t[HD];
            rms_(yw + b * HD, HD, G->qnw, t); rot1(t, S->gc[w], S->gs[w], q[w][b]);
            rms_(yw + D + b * HD, HD, G->knw, t); rot1(t, S->gc[w], S->gs[w], Kr + b * HD);
            memcpy(Vr + b * HD, yw + 2 * D + b * HD, sizeof(float) * HD);
        }
        a->gM[k][i % cap] = 1.f / (1.f + expf(-(dotv(hn + w * D, G->kmW, D) + G->kmb[0])));
    }
    float *sc = a->score;
    for (int w = 0; w < T; w++) {
        int i = i0 + w, lo = i - a->win + 1; if (lo < first) lo = first;
        int n = i - lo + 1;
        for (int j = 0; j < n; j++) {
            int p = lo + j; const float *kj = a->gK[k] + (size_t)(p % cap) * NH * HD; float mj = a->gM[k][p % cap], di = (float)(i - p);
            for (int b = 0; b < NH; b++) sc[(size_t)b * stride + j] = dotv(q[w][b], kj + b * HD, HD) / 8.f - G->slope[b] * di + G->kb[b] * mj;
        }
        for (int b = 0; b < NH; b++) softmax(sc + (size_t)b * stride, n);
        float *o = out + w * D; memset(o, 0, sizeof(float) * D);
        for (int j = 0; j < n; j++) {
            const float *vj = a->gV[k] + (size_t)((lo + j) % cap) * NH * HD;
            for (int b = 0; b < NH; b++) { float p = sc[(size_t)b * stride + j]; if (p > 1e-9f) axpy(o + b * HD, p, vj + b * HD, HD); }
        }
    }
    mm(pn, out, T, &G->Wo, hn);
    for (int w = 0; w < T; w++) addv(h + w * D, hn + w * D, D);
}

static void group_probs(const nat *a, int q, const float *z, float *out) {
    const readout *H = &a->m->q[q];
    float tmp[MAX_COLS], g[MAX_COLS + 1]; int o = 0;
    for (int k = 0; k < H->ngi; k++) tmp[o++] = 1.f / (1.f + expf(-z[H->gi[k]]));
    for (int k = 0; k < H->ng; k++) {
        int n = H->ngg[k];
        for (int j = 0; j < n; j++) g[j] = z[H->gg[k][j]];
        g[n] = 0.f; softmax(g, n + 1);
        for (int j = 0; j < n; j++) tmp[o++] = g[j];
    }
    for (int c = 0; c < H->n; c++) out[c] = tmp[H->gp[c]];
}
static inline void masked(const nat *a, int q, float *z) { const readout *H = &a->m->q[q]; for (int c = 0; c < H->n; c++) if (!a->qvalid[q][c]) z[c] = NEG; }

static void read_q(const nat *a, const float *in, int T, int q, float *zt, float *out, int tot) {
    const nat_model *m = a->m; int n = m->q[q].n;
    mm(a->panel, in, T, &m->q[q].M, zt);
    for (int w = 0; w < T; w++) { float *z = zt + w * n; addv(z, m->q[q].b, n); masked(a, q, z); group_probs(a, q, z, out + (size_t)w * tot + m->ooff[q]); }
}

static inline int clamp_row(int v, const tens *t) { return v < 0 ? 0 : v >= t->d[0] ? t->d[0] - 1 : v; }

static void block(nat *a, const nat_word *ws, int T, float *out) {
    const nat_model *m = a->m; rec_t *r = a->r; scr_t *S = a->scr; float *pn = a->panel; int i0 = r->n, tot = m->total;
    float *e = a->act[0], *u = a->act[1], *h = a->act[2], *mean = a->act[3], *last = a->act[4], *hf = a->act[5], *hd = a->act[6], *ro = a->act[7], *hn = a->act[8], *tm = a->act[9];
    /* ---- word input */
    for (int w = 0; w < T; w++) {
        const nat_word *x = ws + w; float Et[D], *mw = mean + w * D, *lw = last + w * D; memset(mw, 0, sizeof(float) * D);
        int np = x->npc > 8 ? 8 : x->npc < 0 ? 0 : x->npc, li = np > 0 ? np - 1 : 0;
        row_f(&m->tokE, clamp_row(x->pcs[li], &m->tokE), lw);
        for (int p = 0; p < np; p++) {
            if (p == li) memcpy(Et, lw, sizeof Et); else row_f(&m->tokE, clamp_row(x->pcs[p], &m->tokE), Et);
            addv(mw, Et, D); addv(mw, m->ppos + p * D, D);
        }
        float dv = 1.f / (float)(np > 1 ? np : 1); for (int j = 0; j < D; j++) mw[j] *= dv;
    }
    mm(pn, mean, T, &m->W1, e); mm(pn, last, T, &m->W2, tm);
    for (int w = 0; w < T; w++) {
        const nat_word *x = ws + w; float *ew = e + w * D;
        addv(ew, m->b1, D); addv(ew, tm + w * D, D); lnorm_(ew, D, m->lnw, m->lnb);
        row_add(&m->src, clamp_row(x->src, &m->src), ew);
        for (int k = 0; k < NFL; k++) row_add(&m->fl[k], clamp_row(x->wfl[k], &m->fl[k]), ew);
        for (int k = 0; k < NFEATS; k++) row_add(&m->feat[k], clamp_row(x->feats[k], &m->feat[k]), ew);
        int pos = i0 + w - r->first;          /* positions count from the word the state began at */
        rot_tab(m->rope_a, pos, S->c[w], S->s[w]); rot_tab(m->rope_b, pos, S->gc[w], S->gs[w]);
    }
    memcpy(u, e, sizeof(float) * (size_t)T * D);
    for (int l = 0; l < NLOC; l++) mixer_a(a, S, l, T, u);
    for (int w = 0; w < T; w++) lnorm_(u + w * D, D, m->locnw, m->locnb);
    memcpy(tm, u, sizeof(float) * (size_t)T * D);
    for (int w = 0; w < T; w++) lnorm_(tm + w * D, D, m->in_nw, m->in_nb);
    mm(pn, tm, T, &m->Winp, h);
    for (int w = 0; w < T; w++) addv(h + w * D, a->binp, D);
    for (int l = 0; l < NBLK; l++) {
        layer(a, S, l, T, h);
        if (l == 1) mixer_b(a, S, 0, T, h);
        if (l == 3) mixer_b(a, S, 1, T, h);
    }
    { tens Dn = {DT_F32, 2, {D, NTD, 0, 0}, a->tD, NULL}, Up = {DT_F32, 2, {NTD, D, 0, 0}, a->tU, NULL};
      mm(pn, h, T, &Dn, S->t1); gelu(S->t1, T * NTD); mm(pn, S->t1, T, &Up, tm);
      for (int w = 0; w < T; w++) { addv(h + w * D, tm + w * D, D); rms_(h + w * D, D, m->onw, hf + w * D); } }
    /* ---- lookback over earlier words */
    int cap = a->cap, first = r->first;
    float *kk = S->t1, *vv = S->t2, *qq = S->t3;
    for (int w = 0; w < T; w++) rms_(hf + w * D, D, m->rnw, hn + w * D);
    mm(pn, hn, T, &m->rk, kk); mm(pn, hn, T, &m->rv, vv);
    for (int w = 0; w < T; w++) {
        int i = i0 + w;
        rms_(kk + w * HD, HD, m->rknw, a->mK + (size_t)(i % cap) * HD);
        memcpy(a->mV + (size_t)(i % cap) * HD, vv + w * HD, sizeof(float) * HD);
        float ml = dotv(hn + w * D, m->rkmW, D) + m->rkmb[0]; a->mB[i % cap] = m->rkb * (ml < 0.f ? ml - log1pf(expf(ml)) : -log1pf(expf(-ml)));   /* log of the logistic */
        memcpy(hd + w * D, hf + w * D, sizeof(float) * D); addv(hd + w * D, m->q0, D);
        rms_(hd + w * D, D, m->rnw, hn + w * D);
    }
    mm(pn, hn, T, &m->rq, qq);
    float *rd = kk;   /* [T, HD] */
    for (int w = 0; w < T; w++) {
        float qn[HD], *rw = rd + w * HD; rms_(qq + w * HD, HD, m->rqnw, qn); memset(rw, 0, sizeof(float) * HD);
        int i = i0 + w, lo = i - a->win; if (lo < first) lo = first; int n = i - lo;
        if (n > 0) {
            float *sc = a->score; sc[0] = 0.f;
            for (int j = 0; j < n; j++) { int p = (lo + j) % cap; sc[j + 1] = dotv(qn, a->mK + (size_t)p * HD, HD) / 8.f + a->mB[p]; }
            softmax(sc, n + 1);
            for (int j = 0; j < n; j++) if (sc[j + 1] > 1e-9f) axpy(rw, sc[j + 1], a->mV + (size_t)((lo + j) % cap) * HD, HD);
        }
    }
    mm(pn, rd, T, &m->ro, ro);
    for (int w = 0; w < T; w++) addv(ro + w * D, hd + w * D, D);
    mm(pn, rd, T, &m->rou, vv);      /* [T, span_beg columns] */
    /* ---- word read-outs */
    static const int WQ[6] = {Q_FRAME, Q_IN, Q_END, Q_AT, Q_IMM, Q_REC};
    for (int k = 0; k < 6; k++) read_q(a, ro, T, WQ[k], S->t3, out, tot);
    { for (int w = 0; w < T; w++) { float *v = tm + w * D; memcpy(v, u + w * D, sizeof(float) * D); addv(v, m->sbrb, D); addv(v, m->sbk0, D); lnorm_(v, D, m->sblnw, m->sblnb); }
      int n = m->q[Q_BEG].n; mm(pn, tm, T, &m->sbM, S->t3);
      for (int w = 0; w < T; w++) { float *z = S->t3 + w * n; addv(z, m->q[Q_BEG].b, n); masked(a, Q_BEG, z); addv(z, vv + w * n, n); group_probs(a, Q_BEG, z, out + (size_t)w * tot + m->ooff[Q_BEG]); } }
    mm(pn, ro, T, &m->shW, S->t3);
    for (int w = 0; w < T; w++) {
        int shn = m->shW.d[1];
        float *sh = S->t3 + w * shn; addv(sh, m->shb, shn); float *ow = out + (size_t)w * tot;
        int o = 0; for (int f = 0; f < NSH; f++) { float g[MAX_COLS]; int n = m->shd[f]; memcpy(g, sh + o, sizeof(float) * (size_t)n); softmax(g, n); memcpy(ow + m->ooff[NQ + f], g, sizeof(float) * (size_t)n); o += n; }
        ow[m->ooff[NQ + NSH]] = dotv(hf + w * D, m->vw1, D);
        ow[m->ooff[NQ + NSH + 1]] = dotv(hf + w * D, m->vw2, D) + m->vb[0];
    }
    /* ---- pooled read-outs: running sums, read at every word */
    float *pz[3] = {a->act[10], a->act[11], a->act[12]}, *rdu = a->act[13], *rds = a->act[14], *rdi = a->act[15];
    for (int k = 0; k < 3; k++) {
        mm(pn, hf, T, &m->P1[k], S->t1);
        for (int w = 0; w < T; w++) addv(S->t1 + w * D, m->p1[k], D);
        gelu(S->t1, T * D);
        mm(pn, S->t1, T, &m->P2[k], pz[k]);
        for (int w = 0; w < T; w++) { addv(pz[k] + w * D, hf + w * D, D); addv(pz[k] + w * D, m->p2[k], D); }
    }
    for (int w = 0; w < T; w++) {
        const nat_word *x = ws + w; int i = i0 + w;
        float wv = dotv(pz[0] + w * D, m->uaw, D) + m->uab[0]; wv = wv < -10.f ? -10.f : wv > 10.f ? 10.f : wv; double wd = (double)expf(wv);
        if (x->ust > 0 || i == first) { r->utt_w = 0; memset(r->utt_z, 0, sizeof r->utt_z); }
        r->utt_w += wd;
        for (int j = 0; j < D; j++) { r->utt_z[j] += wd * (double)pz[0][w * D + j]; rdu[w * D + j] = (float)(r->utt_z[j] / r->utt_w); }
        int sp = x->spk < 0 ? 0 : x->spk > 9 ? 9 : x->spk;
        if (x->spk > 0) { r->spk_n[sp] += 1.0; double nn = r->spk_n[sp]; for (int j = 0; j < D; j++) { r->spk_z[sp][j] += (double)pz[1][w * D + j]; rds[w * D + j] = (float)(r->spk_z[sp][j] / nn); } }
        else memset(rds + w * D, 0, sizeof(float) * D);
        double nn = (double)(i - first + 1);
        for (int j = 0; j < D; j++) { r->item_z[j] += (double)pz[2][w * D + j]; rdi[w * D + j] = (float)(r->item_z[j] / nn); }
    }
    read_q(a, rdu, T, Q_UTT, S->t3, out, tot); read_q(a, rds, T, Q_SPK, S->t3, out, tot); read_q(a, rdi, T, Q_ITEM, S->t3, out, tot);
    r->n = i0 + T;
    maybe_snap(a);
}
void nat_steps(nat *a, const nat_word *w, int T, float *out) {
    blas_guard g = blas_enter();
    int tot = a->m->total, t = 0;
    while (t < T) {
        int b = T - t < BLOCK ? T - t : BLOCK;
        if (a->snap_on) { int to = SNAP_EVERY - a->r->n % SNAP_EVERY; if (b > to) b = to; }   /* blocks end on snapshot points */
        block(a, w + t, b, out + (size_t)t * tot); t += b;
    }
    blas_leave(g);
}
void nat_set_snapshots(nat *a, int on) { a->snap_on = on; }

#ifdef NAT_AVX2
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif
