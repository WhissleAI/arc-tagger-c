/* See onnx_file.h. Field numbers are those of onnx.proto (ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto,
 * ValueInfoProto, TypeProto, TensorShapeProto). */
#include "onnx_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ wire format */
typedef struct { const uint8_t *p, *e; int bad; } pb;

static uint64_t pb_varint(pb *r) {
    uint64_t v = 0;
    for (int s = 0; s < 64; s += 7) {
        if (r->p >= r->e) { r->bad = 1; return 0; }
        uint8_t b = *r->p++;
        v |= (uint64_t)(b & 0x7F) << s;
        if (!(b & 0x80)) return v;
    }
    r->bad = 1;
    return 0;
}
/* next field: 1, or 0 at the end / on bad data */
static int pb_next(pb *r, uint32_t *fno, int *wt) {
    if (r->bad || r->p >= r->e) return 0;
    uint64_t k = pb_varint(r);
    if (r->bad) return 0;
    *fno = (uint32_t)(k >> 3); *wt = (int)(k & 7);
    if (*fno == 0) { r->bad = 1; return 0; }
    return 1;
}
static int pb_bytes(pb *r, pb *sub) {
    uint64_t n = pb_varint(r);
    if (r->bad || n > (uint64_t)(r->e - r->p)) { r->bad = 1; return -1; }
    sub->p = r->p; sub->e = r->p + n; sub->bad = 0; r->p += n;
    return 0;
}
static void pb_skip(pb *r, int wt) {
    pb s;
    switch (wt) {
    case 0: pb_varint(r); break;
    case 1: if (r->e - r->p < 8) r->bad = 1; else r->p += 8; break;
    case 2: pb_bytes(r, &s); break;
    case 5: if (r->e - r->p < 4) r->bad = 1; else r->p += 4; break;
    default: r->bad = 1;
    }
}

/* ------------------------------------------------------------------ storage: strings and small arrays, never moved */
#define CHUNK (1 << 16)
static void *arena(ox_graph *g, size_t n) {
    n = (n + 7) & ~(size_t)7;
    if (n > CHUNK / 4) {                                   /* large: its own block */
        void *b = malloc(n);
        if (!b) return NULL;
        if (g->slots_used == g->slots_cap) {
            size_t c = g->slots_cap ? 2 * g->slots_cap : 64;
            const char **s = (const char **)realloc(g->slots, c * sizeof *s);
            if (!s) { free(b); return NULL; }
            g->slots = s; g->slots_cap = c;
        }
        g->slots[g->slots_used++] = (const char *)b;
        return b;
    }
    if (!g->pool || g->pool_used + n > g->pool_cap) {
        char *b = (char *)malloc(CHUNK);
        if (!b) return NULL;
        if (g->slots_used == g->slots_cap) {
            size_t c = g->slots_cap ? 2 * g->slots_cap : 64;
            const char **s = (const char **)realloc(g->slots, c * sizeof *s);
            if (!s) { free(b); return NULL; }
            g->slots = s; g->slots_cap = c;
        }
        g->slots[g->slots_used++] = b;
        g->pool = b; g->pool_used = 0; g->pool_cap = CHUNK;
    }
    void *p = g->pool + g->pool_used; g->pool_used += n;
    return p;
}
static const char *str_of(ox_graph *g, const pb *s) {
    size_t n = (size_t)(s->e - s->p);
    char *c = (char *)arena(g, n + 1);
    if (!c) return NULL;
    memcpy(c, s->p, n); c[n] = 0;
    return c;
}

typedef struct { ox_graph *g; char *err; size_t errlen; int failed; } ctx;
static void fail(ctx *c, const char *msg) {
    if (c->failed++) return;
    if (c->err && c->errlen) snprintf(c->err, c->errlen, "%s", msg);
}

/* ------------------------------------------------------------------ messages */
static int parse_tensor(ctx *c, pb r, ox_tensor *t) {
    memset(t, 0, sizeof *t); t->name = "";
    uint32_t f; int wt; pb s;
    int have_data = 0;
    while (pb_next(&r, &f, &wt)) {
        if (f == 1) {                                       /* dims: packed or not */
            if (wt == 2) {
                if (pb_bytes(&r, &s)) break;
                while (s.p < s.e && !s.bad) { int64_t v = (int64_t)pb_varint(&s); if (t->nd < OX_MAX_DIMS) t->d[t->nd] = v; t->nd++; }
                if (s.bad) r.bad = 1;
            } else if (wt == 0) { int64_t v = (int64_t)pb_varint(&r); if (t->nd < OX_MAX_DIMS) t->d[t->nd] = v; t->nd++; }
            else pb_skip(&r, wt);
        } else if (f == 2 && wt == 0) t->dt = (int)pb_varint(&r);
        else if (f == 8 && wt == 2) { if (pb_bytes(&r, &s)) break; t->name = str_of(c->g, &s); if (!t->name) { fail(c, "out of memory"); return -1; } }
        else if (f == 9 && wt == 2) { if (pb_bytes(&r, &s)) break; t->data = s.p; t->len = (size_t)(s.e - s.p); t->enc = 0; have_data = 1; }
        else if ((f == 4 || f == 5 || f == 7 || f == 10) && wt == 2 && !have_data) {
            if (pb_bytes(&r, &s)) break;
            t->data = s.p; t->len = (size_t)(s.e - s.p); t->enc = f == 4 ? 1 : f == 10 ? 3 : 2; have_data = 1;
        } else if (f == 14 && wt == 0) { if (pb_varint(&r) != 0) { fail(c, "onnx: external tensor data is not supported"); return -1; } }
        else pb_skip(&r, wt);
    }
    if (r.bad) { fail(c, "onnx: malformed tensor"); return -1; }
    if (t->nd > OX_MAX_DIMS) { fail(c, "onnx: tensor rank too large"); return -1; }
    int64_t n = 1;
    for (int k = 0; k < t->nd; k++) {
        if (t->d[k] < 0 || (t->d[k] && n > INT64_MAX / t->d[k])) { fail(c, "onnx: bad tensor dimensions"); return -1; }
        n *= t->d[k];
    }
    t->count = n;
    return 0;
}

static int parse_node(ctx *c, pb r, ox_node *n) {
    memset(n, 0, sizeof *n); n->op = "";
    uint32_t f; int wt; pb s;
    /* count inputs / outputs first, then fill */
    pb r2 = r; int nin = 0, nout = 0;
    while (pb_next(&r2, &f, &wt)) { if (f == 1 && wt == 2) nin++; else if (f == 2 && wt == 2) nout++; pb_skip(&r2, wt); }
    if (r2.bad) { fail(c, "onnx: malformed node"); return -1; }
    n->in = (const char **)arena(c->g, sizeof(char *) * (size_t)(nin + 1));
    n->out = (const char **)arena(c->g, sizeof(char *) * (size_t)(nout + 1));
    if (!n->in || !n->out) { fail(c, "out of memory"); return -1; }
    while (pb_next(&r, &f, &wt)) {
        if ((f == 1 || f == 2 || f == 4) && wt == 2) {
            if (pb_bytes(&r, &s)) break;
            const char *v = str_of(c->g, &s);
            if (!v) { fail(c, "out of memory"); return -1; }
            if (f == 1) n->in[n->nin++] = v; else if (f == 2) n->out[n->nout++] = v; else n->op = v;
        } else if (f == 5 && wt == 2) {                    /* attribute */
            pb a; if (pb_bytes(&r, &a)) break;
            const char *an = ""; int64_t iv = 0; int has_i = 0; pb tv = {0, 0, 0}; int has_t = 0;
            uint32_t af; int awt; pb as;
            while (pb_next(&a, &af, &awt)) {
                if (af == 1 && awt == 2) { if (pb_bytes(&a, &as)) break; an = str_of(c->g, &as); if (!an) { fail(c, "out of memory"); return -1; } }
                else if (af == 3 && awt == 0) { iv = (int64_t)pb_varint(&a); has_i = 1; }
                else if (af == 5 && awt == 2) { if (pb_bytes(&a, &tv)) break; has_t = 1; }
                else pb_skip(&a, awt);
            }
            if (a.bad) { r.bad = 1; break; }
            if (!strcmp(an, "axis") && has_i) { n->axis = iv; n->has_axis = 1; }
            if (!strcmp(an, "value") && has_t) { if (parse_tensor(c, tv, &n->value)) return -1; n->has_value = 1; }
        } else pb_skip(&r, wt);
    }
    if (r.bad) { fail(c, "onnx: malformed node"); return -1; }
    return 0;
}

static int parse_output(ctx *c, pb r, ox_output *o) {
    memset(o, 0, sizeof *o); o->name = "";
    uint32_t f; int wt; pb s;
    while (pb_next(&r, &f, &wt)) {
        if (f == 1 && wt == 2) { if (pb_bytes(&r, &s)) break; o->name = str_of(c->g, &s); if (!o->name) { fail(c, "out of memory"); return -1; } }
        else if (f == 2 && wt == 2) {                      /* TypeProto.tensor_type(1).shape(2).dim(1).dim_value(1) */
            pb ty; if (pb_bytes(&r, &ty)) break;
            uint32_t f1; int w1;
            while (pb_next(&ty, &f1, &w1)) {
                if (f1 != 1 || w1 != 2) { pb_skip(&ty, w1); continue; }
                pb tt; if (pb_bytes(&ty, &tt)) break;
                uint32_t f2; int w2;
                while (pb_next(&tt, &f2, &w2)) {
                    if (f2 != 2 || w2 != 2) { pb_skip(&tt, w2); continue; }
                    pb sh; if (pb_bytes(&tt, &sh)) break;
                    uint32_t f3; int w3;
                    while (pb_next(&sh, &f3, &w3)) {
                        if (f3 != 1 || w3 != 2) { pb_skip(&sh, w3); continue; }
                        pb dm; if (pb_bytes(&sh, &dm)) break;
                        int64_t v = -1; uint32_t f4; int w4;
                        while (pb_next(&dm, &f4, &w4)) { if (f4 == 1 && w4 == 0) v = (int64_t)pb_varint(&dm); else pb_skip(&dm, w4); }
                        if (o->nd < OX_MAX_DIMS) o->d[o->nd] = v;
                        o->nd++;
                    }
                }
            }
            if (ty.bad) r.bad = 1;
        } else pb_skip(&r, wt);
    }
    if (r.bad) { fail(c, "onnx: malformed graph output"); return -1; }
    if (o->nd > OX_MAX_DIMS) o->nd = OX_MAX_DIMS;
    return 0;
}

#define GROW(arr, n, cap) do { if ((n) == (cap)) { int nc_ = (cap) ? 2 * (cap) : 256; void *p_ = realloc((arr), sizeof *(arr) * (size_t)nc_); \
    if (!p_) { fail(&c, "out of memory"); goto bad; } (arr) = p_; (cap) = nc_; } } while (0)

int ox_parse(ox_graph *g, const uint8_t *buf, size_t size, char *err, size_t errlen) {
    memset(g, 0, sizeof *g);
    ctx c = {g, err, errlen, 0};
    pb r = {buf, buf + size, 0}, gr = {0, 0, 0};
    uint32_t f; int wt; int have = 0;
    while (pb_next(&r, &f, &wt)) {
        if (f == 7 && wt == 2) { if (pb_bytes(&r, &gr)) break; have = 1; }
        else pb_skip(&r, wt);
    }
    if (r.bad || !have) { fail(&c, have ? "onnx: malformed file" : "onnx: no graph in the file (not an ONNX model?)"); goto bad; }
    int ci = 0, cn = 0, co = 0;
    while (pb_next(&gr, &f, &wt)) {
        if (wt != 2) { pb_skip(&gr, wt); continue; }
        pb s; if (pb_bytes(&gr, &s)) break;
        if (f == 5) { GROW(g->init, g->ninit, ci); if (parse_tensor(&c, s, &g->init[g->ninit])) goto bad; g->ninit++; }
        else if (f == 1) { GROW(g->node, g->nnode, cn); if (parse_node(&c, s, &g->node[g->nnode])) goto bad; g->nnode++; }
        else if (f == 12) { GROW(g->out, g->nout, co); if (parse_output(&c, s, &g->out[g->nout])) goto bad; g->nout++; }
    }
    if (gr.bad) { fail(&c, "onnx: malformed graph"); goto bad; }
    return 0;
bad:
    ox_free(g);
    return -1;
}

int ox_load(ox_graph *g, const char *path, char *err, size_t errlen) {
    memset(g, 0, sizeof *g);
    FILE *fp = fopen(path, "rb");
    if (!fp) { if (err && errlen) snprintf(err, errlen, "cannot open %s", path); return -1; }
    long n = -1;
    if (!fseek(fp, 0, SEEK_END)) n = ftell(fp);
    if (n <= 0 || fseek(fp, 0, SEEK_SET)) { fclose(fp); if (err && errlen) snprintf(err, errlen, "%s: empty or unreadable", path); return -1; }
    uint8_t *b = (uint8_t *)malloc((size_t)n);
    if (!b) { fclose(fp); if (err && errlen) snprintf(err, errlen, "out of memory"); return -1; }
    size_t got = fread(b, 1, (size_t)n, fp);
    fclose(fp);
    if (got != (size_t)n) { free(b); if (err && errlen) snprintf(err, errlen, "%s: read error", path); return -1; }
    if (ox_parse(g, b, (size_t)n, err, errlen)) { free(b); return -1; }
    g->buf = b; g->size = (size_t)n;
    return 0;
}

void ox_free(ox_graph *g) {
    if (!g) return;
    for (size_t i = 0; i < g->slots_used; i++) free((void *)g->slots[i]);
    free(g->slots); free(g->init); free(g->node); free(g->out); free(g->buf);
    memset(g, 0, sizeof *g);
}

/* ------------------------------------------------------------------ lookups */
const ox_tensor *ox_init(const ox_graph *g, const char *name) {
    for (int i = 0; i < g->ninit; i++) if (!strcmp(g->init[i].name, name)) return &g->init[i];
    return NULL;
}
const ox_node *ox_producer(const ox_graph *g, const char *name) {
    for (int i = 0; i < g->nnode; i++) for (int k = 0; k < g->node[i].nout; k++) if (!strcmp(g->node[i].out[k], name)) return &g->node[i];
    return NULL;
}
const ox_tensor *ox_value(const ox_graph *g, const char *name) {
    for (int depth = 0; depth < 16; depth++) {
        const ox_tensor *t = ox_init(g, name);
        if (t) return t;
        const ox_node *n = ox_producer(g, name);
        if (!n) return NULL;
        if (!strcmp(n->op, "Constant")) return n->has_value ? &n->value : NULL;
        if (strcmp(n->op, "Identity") || n->nin != 1) return NULL;
        name = n->in[0];
    }
    return NULL;
}
const ox_node *ox_only_consumer(const ox_graph *g, const char *name, const char *op) {
    const ox_node *hit = NULL;
    for (int i = 0; i < g->nnode; i++) {
        const ox_node *n = &g->node[i];
        if (strcmp(n->op, op)) continue;
        for (int k = 0; k < n->nin; k++) if (!strcmp(n->in[k], name)) { if (hit && hit != n) return NULL; hit = n; break; }
    }
    return hit;
}
int ox_node_index(const ox_graph *g, const ox_node *n) { return (int)(n - g->node); }

/* ------------------------------------------------------------------ element conversion */
static int is_int_type(int dt) { return dt == OX_UINT8 || dt == OX_INT8 || dt == OX_INT32 || dt == OX_INT64 || dt == OX_BOOL; }
static int raw_size(int dt) { return dt == OX_FLOAT || dt == OX_INT32 ? 4 : dt == OX_INT64 || dt == OX_DOUBLE ? 8 : dt == OX_UINT8 || dt == OX_INT8 || dt == OX_BOOL ? 1 : 0; }

static int64_t raw_int(const uint8_t *p, int dt) {
    switch (dt) {
    case OX_UINT8: case OX_BOOL: return p[0];
    case OX_INT8: return (int8_t)p[0];
    case OX_INT32: { int32_t v; memcpy(&v, p, 4); return v; }
    default: { int64_t v; memcpy(&v, p, 8); return v; }
    }
}
static void put_int(void *dst, int dst_type, int64_t i, int64_t v) {
    if (dst_type == OX_INT64) ((int64_t *)dst)[i] = v;
    else if (dst_type == OX_INT8) ((int8_t *)dst)[i] = (int8_t)v;
    else ((uint8_t *)dst)[i] = (uint8_t)v;
}

int ox_read(const ox_tensor *t, int dst_type, void *dst, int64_t count) {
    if (count != t->count) return -1;
    int want_float = dst_type == OX_FLOAT;
    if (want_float != (t->dt == OX_FLOAT || t->dt == OX_DOUBLE) || (!want_float && !is_int_type(t->dt))) return -1;
    if (t->enc == 0) {                                      /* raw_data: little-endian, packed */
        int es = raw_size(t->dt);
        if (!es || t->len != (size_t)count * (size_t)es) return -1;
        if (t->dt == OX_FLOAT) { memcpy(dst, t->data, t->len); return 0; }
        if (t->dt == OX_DOUBLE) { for (int64_t i = 0; i < count; i++) { double v; memcpy(&v, t->data + 8 * i, 8); ((float *)dst)[i] = (float)v; } return 0; }
        for (int64_t i = 0; i < count; i++) put_int(dst, dst_type, i, raw_int(t->data + (size_t)i * (size_t)es, t->dt));
        return 0;
    }
    if (t->enc == 1) { if (t->dt != OX_FLOAT || t->len != (size_t)count * 4) return -1; memcpy(dst, t->data, t->len); return 0; }
    if (t->enc == 3) {
        if (t->len != (size_t)count * 8) return -1;
        for (int64_t i = 0; i < count; i++) { double v; memcpy(&v, t->data + 8 * i, 8); ((float *)dst)[i] = (float)v; }
        return 0;
    }
    pb r = {t->data, t->data + t->len, 0};                   /* packed varints */
    for (int64_t i = 0; i < count; i++) {
        uint64_t v = pb_varint(&r);
        if (r.bad) return -1;
        int64_t s = t->dt == OX_INT32 ? (int64_t)(int32_t)(uint32_t)v : (int64_t)v;
        put_int(dst, dst_type, i, s);
    }
    return r.p == r.e ? 0 : -1;
}
