/* arctag: Arc Tagger C library, native runtime (no ONNX Runtime). Features, stream bookkeeping, decoding and JSON live here; the
 * forward pass is atn.c. Decoding = the Arc Tagger demo engine's T0 path (see README). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "arctag.h"
#include "tokenizer.h"
#include "cJSON.h"
#include "atn.h"
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARCTAG_VERSION "0.2.0"
#define NQ 10
#define P_MAX 8
#define NFEAT 8
#define MAX_FAM 8
#define MAX_OUT 32
#define NSLOT 16                /* speaker slots tracked by the features */
#define CHUNK 64                /* words stepped between row compactions in arctag_poll */
#define REWIND_ROWS 96          /* rows kept beyond the decode window: a revision can reach this far back and still decode */
#define MAX_WORD_BYTES 1024     /* longer words are tokenized by their head and tail */
#define MAX_BACKLOG 65536       /* words pushed but not yet polled */

static const char *QN[NQ] = {"frame", "span_beg", "span_in", "span_end", "ev_at", "ev_imm", "ev_rec", "utt", "speaker", "item"};
enum { Q_FRAME, Q_BEG, Q_IN, Q_END, Q_EVAT, Q_EVIMM, Q_EVREC, Q_UTT, Q_SPK, Q_ITEM };

/* ------------------------------------------------------------------ allocation
 * Every allocation goes through here or through cJSON's hooks (set below), which raise a per-thread flag on failure. Each public
 * call clears the flag on entry and, if it was raised, frees its result and reports "out of memory". So an out-of-memory path only
 * has to avoid dereferencing NULL; it never returns a JSON document with pieces silently missing. */
static __thread int tls_oom;
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) tls_oom = 1; return p; }
static void *xcalloc(size_t n, size_t z) { void *p = calloc(n ? n : 1, z ? z : 1); if (!p) tls_oom = 1; return p; }
static void *xrealloc(void *o, size_t n) { void *p = realloc(o, n ? n : 1); if (!p) tls_oom = 1; return p; }
static void hook_free(void *p) { free(p); }
static void install_hooks(void) { cJSON_Hooks k = {xmalloc, hook_free}; cJSON_InitHooks(&k); }
static void api_enter(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, install_hooks);
    tls_oom = 0;
}

/* ------------------------------------------------------------------ small utils */
static char *xstrdup(const char *s) { if (!s) s = ""; size_t n = strlen(s) + 1; char *c = (char *)xmalloc(n); if (c) memcpy(c, s, n); return c; }
static double r6(double x) { return round(x * 1e6) / 1e6; }
static int starts_with(const char *s, const char *p) { return strncmp(s, p, strlen(p)) == 0; }

/* ------------------------------------------------------------------ model metadata */
typedef struct { char *name; int dom; double *thr[NQ]; unsigned char *valid[NQ]; double head_thr[MAX_FAM]; } profile_t;
typedef struct { char *name, *out_t, *out_b; int ntags; char **tags; int use; } fam_t;

typedef struct {
    int nc[NQ]; char **cols[NQ];
    int *beg_of_in, *is_norm;
    int nfam; fam_t fam[MAX_FAM];
    int nsg; char **state_groups;
    int nprof; profile_t *prof;
    int nalias; char **alias_k; int *alias_v;
    int default_prof;
    double gap_e[16], dur_e[16], conf_e[16]; int n_gap, n_dur, n_conf;
    int nasr; char **asr; int default_asr;
    int max_ctx, span_win;
} meta_t;

/* ------------------------------------------------------------------ words / stream */
typedef struct {
    char *w; double t0, t1, conf; int has_t0, has_t1, has_conf, final; char *spk;
    int npc; int32_t pcs[P_MAX]; int hd, all;              /* pieces, has a digit, all digits */
    /* features: a pure function of this word and the words before it, so computed once at push (see word_feat) */
    int gap, dur, conf_b, chg, sp, fd, fg, spk_any;
    double last_t1; int last_has_t1;                       /* after this word: the latest end time */
    int last_spk_i;                                        /* after this word: index of the latest labelled word, or -1 */
    int nslot;                                             /* after this word: speaker slots in use */
} word_t;

typedef struct { int q, c, i; } evkey_t;
typedef struct { int start, end; } uspan_t;

struct arctag {
    atn_model *nm; atn *one, *st;
    int nat[MAX_OUT], ooff[MAX_OUT], total;
    meta_t m; arctok_t *tok;
    int prof, asr, max_ctx, stream_ctx;
    char err[512];
    /* stream. Word indices are global and never reused; only a bounded suffix of the session is held:
     * words [wbase, n), rows of model output [rbase, sdone). See the README's "Long sessions". */
    word_t *words; int wbase, n, cap;
    float *rows; int rcap, rbase, sdone, sdirty;
    char *slot[NSLOT]; char *old_spk;                      /* speaker slot names; the label of the newest dropped labelled word */
    int *bounds; int nb, bcap, bfloor;                     /* utterance ends (exclusive), increasing, >= wbase; bfloor: the newest dropped one */
    uspan_t *pend; int np, pcap;                           /* closed utterances not yet reported */
    evkey_t *keys; int nk, kcap;
    int lo;                                                /* lowest word index whose events are not yet checked */
    /* output names */
    int nout; char *out_names[MAX_OUT];
    int qo[NQ];                                            /* output index of each read-out */
};

static void set_err(arctag_t *h, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); vsnprintf(h->err, sizeof h->err, fmt, ap); va_end(ap);
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    long n = -1;
    if (!fseek(f, 0, SEEK_END)) n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
    char *b = (char *)xmalloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); b = NULL; }
    fclose(f); if (b) b[n] = 0;
    return b;
}

static char **str_array(cJSON *a, int *n) {
    int c = cJSON_GetArraySize(a);
    char **o = (char **)xcalloc((size_t)c + 1, sizeof(char *));
    *n = 0;
    if (!o) return NULL;
    for (cJSON *x = a ? a->child : NULL; x && *n < c; x = x->next) {
        if (!(o[*n] = xstrdup(cJSON_IsString(x) ? x->valuestring : ""))) break;
        (*n)++;
    }
    return o;
}

static int num_array(cJSON *a, double *o, int cap) {
    int i = 0;
    for (cJSON *x = a ? a->child : NULL; x && i < cap; x = x->next) o[i++] = cJSON_IsNumber(x) ? x->valuedouble : 0;
    return i;
}

static int find_str(char **arr, int n, const char *s) {
    for (int i = 0; i < n; i++) if (!strcmp(arr[i], s)) return i;
    return -1;
}

static int load_meta(arctag_t *h, const char *path) {
    char *txt = read_file(path);
    if (!txt) { set_err(h, "cannot read %s", path); return -1; }
    cJSON *j = cJSON_Parse(txt); free(txt);
    if (!j) { set_err(h, "invalid JSON in %s", path); return -1; }
    meta_t *m = &h->m;
#define META_FAIL(...) do { set_err(h, __VA_ARGS__); cJSON_Delete(j); return -1; } while (0)
    cJSON *ro = cJSON_GetObjectItem(j, "readouts");
    for (int q = 0; q < NQ; q++) {
        cJSON *a = cJSON_GetObjectItem(ro, QN[q]);
        if (!cJSON_IsArray(a)) META_FAIL("meta: readouts.%s missing", QN[q]);
        m->cols[q] = str_array(a, &m->nc[q]);
        if (!m->cols[q] || m->nc[q] < 1) META_FAIL("meta: readouts.%s is empty", QN[q]);
    }
    int nin = m->nc[Q_IN];
    m->beg_of_in = (int *)xcalloc((size_t)nin, sizeof(int)); m->is_norm = (int *)xcalloc((size_t)nin, sizeof(int));
    if (!m->beg_of_in || !m->is_norm) META_FAIL("out of memory");
    double tmp[1024];
    int k = num_array(cJSON_GetObjectItem(j, "beg_of_in"), tmp, 1024);
    for (int i = 0; i < nin; i++) m->beg_of_in[i] = i < k ? (int)tmp[i] : -1;
    k = num_array(cJSON_GetObjectItem(j, "is_norm"), tmp, 1024);
    for (int i = 0; i < nin; i++) m->is_norm[i] = i < k ? (int)tmp[i] : 0;
    cJSON *sh = cJSON_GetObjectItem(j, "span_head");
    for (cJSON *f = sh ? sh->child : NULL; f && m->nfam < MAX_FAM; f = f->next) {
        cJSON *fn = cJSON_GetObjectItem(f, "name"), *ft = cJSON_GetObjectItem(f, "out_t"), *fb = cJSON_GetObjectItem(f, "out_b");
        if (!cJSON_IsString(fn) || !cJSON_IsString(ft) || !cJSON_IsString(fb)) META_FAIL("meta: bad span_head entry");
        fam_t *F = &m->fam[m->nfam++];
        F->name = xstrdup(fn->valuestring); F->out_t = xstrdup(ft->valuestring); F->out_b = xstrdup(fb->valuestring);
        F->tags = str_array(cJSON_GetObjectItem(f, "tags"), &F->ntags);
        if (!F->name || !F->out_t || !F->out_b || !F->tags) META_FAIL("out of memory");
        F->use = cJSON_IsTrue(cJSON_GetObjectItem(f, "use"));
    }
    m->state_groups = str_array(cJSON_GetObjectItem(j, "state_groups"), &m->nsg);
    if (!m->state_groups) META_FAIL("out of memory");
    m->n_gap = num_array(cJSON_GetObjectItem(j, "gap_edges"), m->gap_e, 16);
    m->n_dur = num_array(cJSON_GetObjectItem(j, "dur_edges"), m->dur_e, 16);
    m->n_conf = num_array(cJSON_GetObjectItem(j, "conf_edges"), m->conf_e, 16);
    m->asr = str_array(cJSON_GetObjectItem(j, "asr_sources"), &m->nasr);
    if (!m->asr || m->nasr < 1) META_FAIL("meta: asr_sources is empty");
    cJSON *da = cJSON_GetObjectItem(j, "default_asr");
    m->default_asr = cJSON_IsString(da) ? find_str(m->asr, m->nasr, da->valuestring) : 0;
    if (m->default_asr < 0) m->default_asr = 0;
    cJSON *x = cJSON_GetObjectItem(j, "max_ctx"); m->max_ctx = cJSON_IsNumber(x) ? x->valueint : 2048;
    if (m->max_ctx < 16 || m->max_ctx > 2048) META_FAIL("meta: max_ctx %d is outside 16..2048", m->max_ctx);
    x = cJSON_GetObjectItem(j, "span_win"); m->span_win = cJSON_IsNumber(x) ? x->valueint : 96;
    if (m->span_win < 1 || m->span_win > 2048) META_FAIL("meta: span_win %d is outside 1..2048", m->span_win);
    cJSON *pr = cJSON_GetObjectItem(j, "profiles");
    if (!cJSON_IsObject(pr) || !pr->child) META_FAIL("meta: profiles missing");
    int npr = cJSON_GetArraySize(pr);
    m->prof = (profile_t *)xcalloc((size_t)npr, sizeof(profile_t));
    if (!m->prof) META_FAIL("out of memory");
    int pi = 0;
    for (cJSON *p = pr->child; p; p = p->next, pi++) {
        profile_t *P = &m->prof[pi];
        cJSON *pd = cJSON_GetObjectItem(p, "dom");
        if (!p->string || !cJSON_IsNumber(pd)) META_FAIL("meta: bad profile entry");
        m->nprof = pi + 1;
        if (!(P->name = xstrdup(p->string))) META_FAIL("out of memory");
        P->dom = pd->valueint;
        cJSON *th = cJSON_GetObjectItem(p, "thr"), *va = cJSON_GetObjectItem(p, "valid"), *ht = cJSON_GetObjectItem(p, "head_thr");
        for (int q = 0; q < NQ; q++) {
            P->thr[q] = (double *)xcalloc((size_t)m->nc[q] + 1, sizeof(double));
            P->valid[q] = (unsigned char *)xcalloc((size_t)m->nc[q] + 1, 1);
            if (!P->thr[q] || !P->valid[q]) META_FAIL("out of memory");
            cJSON *a = cJSON_GetObjectItem(th, QN[q]), *b = cJSON_GetObjectItem(va, QN[q]);
            int c = 0;
            for (cJSON *e = a ? a->child : NULL; e && c < m->nc[q]; e = e->next) P->thr[q][c++] = cJSON_IsNumber(e) ? e->valuedouble : 0;
            c = 0;
            for (cJSON *e = b ? b->child : NULL; e && c < m->nc[q]; e = e->next) P->valid[q][c++] = (unsigned char)(cJSON_IsNumber(e) && e->valuedouble != 0);
        }
        for (int f = 0; f < m->nfam; f++) {
            cJSON *v = cJSON_GetObjectItem(ht, m->fam[f].name);
            P->head_thr[f] = cJSON_IsNumber(v) ? v->valuedouble : 0.5;
        }
    }
    cJSON *al = cJSON_GetObjectItem(j, "aliases");
    int nal = cJSON_IsObject(al) ? cJSON_GetArraySize(al) : 0, ai = 0;
    m->alias_k = (char **)xcalloc((size_t)nal + 1, sizeof(char *)); m->alias_v = (int *)xcalloc((size_t)nal + 1, sizeof(int));
    if (!m->alias_k || !m->alias_v) META_FAIL("out of memory");
    for (cJSON *a = nal ? al->child : NULL; a; a = a->next, ai++) {
        if (!a->string || !(m->alias_k[ai] = xstrdup(a->string))) META_FAIL("meta: bad alias");
        m->nalias = ai + 1;
        m->alias_v[ai] = 0;
        for (int p = 0; p < m->nprof; p++) if (cJSON_IsString(a) && !strcmp(m->prof[p].name, a->valuestring)) m->alias_v[ai] = p;
    }
    cJSON *dp = cJSON_GetObjectItem(j, "default_profile");
    m->default_prof = 0;
    for (int p = 0; p < m->nprof; p++) if (cJSON_IsString(dp) && !strcmp(m->prof[p].name, dp->valuestring)) m->default_prof = p;
#undef META_FAIL
    cJSON_Delete(j);
    return 0;
}

/* also frees a partly loaded meta: only filled entries are counted */
static void free_meta(meta_t *m) {
    for (int q = 0; q < NQ; q++) { for (int c = 0; c < m->nc[q]; c++) free(m->cols[q][c]); free(m->cols[q]); }
    free(m->beg_of_in); free(m->is_norm);
    for (int f = 0; f < m->nfam; f++) {
        fam_t *F = &m->fam[f]; free(F->name); free(F->out_t); free(F->out_b);
        for (int t = 0; t < F->ntags; t++) free(F->tags[t]);
        free(F->tags);
    }
    for (int i = 0; i < m->nsg; i++) free(m->state_groups[i]);
    free(m->state_groups);
    for (int i = 0; i < m->nasr; i++) free(m->asr[i]);
    free(m->asr);
    if (m->prof) for (int p = 0; p < m->nprof; p++) {
        free(m->prof[p].name);
        for (int q = 0; q < NQ; q++) { free(m->prof[p].thr[q]); free(m->prof[p].valid[q]); }
    }
    free(m->prof);
    for (int i = 0; i < m->nalias; i++) free(m->alias_k[i]);
    free(m->alias_k); free(m->alias_v);
}

/* ------------------------------------------------------------------ model output views */
typedef struct {
    int W; float *own; float *o[MAX_OUT]; int dim[MAX_OUT], st[MAX_OUT];   /* o[i][row * st[i] + c] */
} run_t;

static void run_free(run_t *r) { free(r->own); memset(r, 0, sizeof *r); }

static int out_index(const arctag_t *h, const char *name) {
    for (int i = 0; i < h->nout; i++) if (!strcmp(h->out_names[i], name)) return i;
    return -1;
}

static int bk(const double *e, int n, double x) { int k = 0; for (int i = 0; i < n; i++) if (e[i] <= x) k++; return 1 + k; }

static int word_ncp(const char *s) { int n = 0; for (; *s; s++) if (((unsigned char)*s & 0xC0) != 0x80) n++; return n; }

/* word flags: has a digit (Python str.isdigit per char), all chars digits */
static void digit_flags(const char *s, int *hd, int *all) {
    size_t n = strlen(s), i = 0; int any = 0, every = n > 0;
    while (i < n) { uint32_t cp; utf8_decode(s, n, &i, &cp); if (uni_is_digit(cp)) any = 1; else every = 0; }
    *hd = any; *all = any && every;
}

static int clamp_d(double x, int a, int b) { return !(x > a) ? a : x > b ? b : (int)x; }   /* also NaN and huge values */

/* The features of word i from the state left by the word before it. slot[] holds the speaker names seen so far; the caller owns
 * them. Same rules as the engine and python/arctag_ref.py. prev_spk: the latest label before this word, or NULL. */
static int word_feat(arctag_t *h, word_t *w, const word_t *prev, int i, const char *prev_spk, char **slot, int dup_names) {
    int have_prev = prev != NULL, prev_has_t1 = prev ? prev->last_has_t1 : 0, nslot = prev ? prev->nslot : 0;
    double prev_t1 = prev ? prev->last_t1 : 0;
    w->gap = w->dur = w->conf_b = w->chg = w->sp = 0;
    if (w->has_t0) {
        w->gap = !prev_has_t1 ? 9 : (w->t0 - prev_t1 < 0 ? 1 : bk(h->m.gap_e, h->m.n_gap, w->t0 - prev_t1));
        if (w->has_t1) { int L = word_ncp(w->w); w->dur = bk(h->m.dur_e, h->m.n_dur, (w->t1 - w->t0) / (L > 1 ? L : 1)); if (w->dur > 7) w->dur = 7; }
    }
    if (w->has_conf) w->conf_b = bk(h->m.conf_e, h->m.n_conf, w->conf);
    w->last_spk_i = prev ? prev->last_spk_i : -1;
    if (w->spk) {
        int s = -1;
        for (int k = 0; k < nslot; k++) if (!strcmp(slot[k], w->spk)) s = k;
        if (s < 0) {
            if (nslot < NSLOT) { slot[nslot] = dup_names ? xstrdup(w->spk) : w->spk; if (!slot[nslot]) return -1; s = nslot++; }
            else s = NSLOT - 1;
        }
        w->sp = s + 1 > 9 ? 9 : s + 1;
        w->chg = !prev_spk ? 0 : (!strcmp(w->spk, prev_spk) ? 1 : 2);
        w->last_spk_i = i;
    }
    if (w->gap && w->has_t0 && w->has_t1) {
        w->fd = clamp_d(rint(w->t1 * 10) - rint(w->t0 * 10) + 1, 1, 20);
        w->fg = !prev_has_t1 ? 20 : clamp_d(rint(w->t0 * 10) - rint(prev_t1 * 10) - 1, 0, 20);
    } else { w->fd = 3; w->fg = have_prev ? 0 : 20; }
    w->last_t1 = w->has_t1 ? w->t1 : prev_t1; w->last_has_t1 = w->has_t1 || prev_has_t1;
    w->nslot = nslot;
    w->spk_any = (prev && prev->spk_any) || w->spk;
    return 0;
}

/* spksrc: 2 once a speaker label has been seen, see the README */
static void word_input(const arctag_t *h, const word_t *w, int spksrc, int ust, atn_word *o) {
    memset(o, 0, sizeof *o);
    for (int p = 0; p < w->npc; p++) o->pcs[p] = w->pcs[p];
    o->npc = w->npc;
    int fl[5] = {w->hd, w->all, w->npc < P_MAX ? w->npc : P_MAX, w->fd, w->fg};
    memcpy(o->wfl, fl, sizeof fl);
    int fe[NFEAT] = {w->gap, w->dur, w->final ? 1 : 2, w->conf_b, w->chg, h->asr, spksrc, 0};
    memcpy(o->feats, fe, sizeof fe);
    o->src = (h->asr >= 0 && !strcmp(h->m.asr[h->asr], "arc_transcriber")) ? 1 : 0;
    o->spk = w->sp; o->ust = ust;
}

static void run_view(const arctag_t *h, run_t *r, float *base, int W) {
    r->W = W;
    for (int i = 0; i < h->nout; i++) { int k = h->nat[i]; r->o[i] = base + h->ooff[i]; r->dim[i] = atn_out_dim(h->nm, k); r->st[i] = h->total; }
}

/* one-shot pass over a text's words. spksrc here is the whole pass's: the graph's input, and what the reference computes. */
static int run_model(arctag_t *h, word_t *ws, int W, const unsigned char *us, run_t *r) {
    memset(r, 0, sizeof *r);
    atn_word *in = (atn_word *)xmalloc(sizeof(atn_word) * (size_t)W);
    r->own = (float *)xmalloc(sizeof(float) * (size_t)W * (size_t)h->total);
    if (!h->one && !tls_oom) h->one = atn_new(h->nm, h->m.prof[h->prof].dom, 2048, 2048);
    if (!in || !r->own || !h->one) { set_err(h, "out of memory"); tls_oom = 1; free(in); free(r->own); r->own = NULL; return -1; }
    char *slot[NSLOT]; int spksrc = 0;
    const char *last_spk = NULL;
    for (int i = 0; i < W; i++) {
        word_feat(h, &ws[i], i ? &ws[i - 1] : NULL, i, last_spk, slot, 0);
        if (ws[i].spk) { last_spk = ws[i].spk; spksrc = 2; }
    }
    for (int i = 0; i < W; i++) word_input(h, &ws[i], spksrc, us[i] ? 1 : 0, &in[i]);
    in[0].ust = 1;
    if (atn_set_domain(h->one, h->m.prof[h->prof].dom)) { set_err(h, "domain out of range"); free(in); free(r->own); r->own = NULL; return -1; }
    atn_set_snapshots(h->one, 0);
    atn_steps(h->one, in, W, r->own);
    free(in);
    run_view(h, r, r->own, W);
    /* a long pass leaves its attention memory committed: give it back rather than keep the high-water mark */
    if (W > 512) { atn_free(h->one); h->one = NULL; }
    return 0;
}

/* probability of read-out q, row (window-relative) r, column c */
static float P(const arctag_t *h, const run_t *r, int q, int row, int c) {
    int oi = h->qo[q];
    return r->o[oi][(size_t)row * r->st[oi] + c];
}

/* ------------------------------------------------------------------ JSON helpers */
static cJSON *jnum(double x) { return cJSON_CreateNumber(r6(x)); }

/* ws[0] is word wb */
static char *join_words(const word_t *ws, int wb, int a, int b) {
    size_t L = 1;
    for (int i = a; i < b; i++) L += strlen(ws[i - wb].w) + 1;
    char *s = (char *)xmalloc(L); size_t k = 0;
    if (!s) return NULL;
    for (int i = a; i < b; i++) { if (i > a) s[k++] = ' '; size_t n = strlen(ws[i - wb].w); memcpy(s + k, ws[i - wb].w, n); k += n; }
    s[k] = 0;
    return s;
}

/* top-k of a read-out among valid columns whose tag starts with prefix, sorted by -p (stable) */
static cJSON *top(const arctag_t *h, const run_t *r, int q, int row, const char *prefix, int k, int *idx_out, float *p_out, int *n_out) {
    const profile_t *pf = &h->m.prof[h->prof];
    int nc = h->m.nc[q];
    int *idx = (int *)xmalloc(sizeof(int) * (size_t)(nc + 1)); int n = 0;
    if (!idx) { if (n_out) *n_out = 0; return cJSON_CreateArray(); }
    for (int c = 0; c < nc; c++) if (pf->valid[q][c] && starts_with(h->m.cols[q][c], prefix)) idx[n++] = c;
    for (int i = 1; i < n; i++) {                                           /* stable insertion sort, descending p */
        int x = idx[i]; float px = P(h, r, q, row, x); int j = i - 1;
        while (j >= 0 && P(h, r, q, row, idx[j]) < px) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = x;
    }
    cJSON *a = cJSON_CreateArray();
    if (n_out) *n_out = 0;
    for (int i = 0; i < n && i < k; i++) {
        int c = idx[i]; float p = P(h, r, q, row, c); double th = pf->thr[q][c];
        cJSON *o = cJSON_CreateObject();
        cJSON_AddStringToObject(o, "tag", h->m.cols[q][c]);
        cJSON_AddItemToObject(o, "p", jnum(p));
        cJSON_AddItemToObject(o, "thr", jnum(th));
        cJSON_AddBoolToObject(o, "on", p >= (float)th);            /* float32 compare (= reference) */
        cJSON_AddItemToArray(a, o);
        if (idx_out) idx_out[i] = c;
        if (p_out) p_out[i] = p;
        if (n_out) (*n_out)++;
    }
    free(idx);
    return a;
}

static void append_all(cJSON *dst, cJSON *src) {
    cJSON *x;
    while ((x = cJSON_DetachItemFromArray(src, 0)) != NULL) cJSON_AddItemToArray(dst, x);
    cJSON_Delete(src);
}

static double jget(cJSON *o, const char *k) { cJSON *x = cJSON_GetObjectItem(o, k); return x ? x->valuedouble : 0; }

static void sort_by_p_desc(cJSON *arr) {                                  /* stable */
    int n = cJSON_GetArraySize(arr);
    cJSON **v = (cJSON **)xmalloc(sizeof(cJSON *) * (size_t)(n + 1));
    if (!v) return;
    for (int i = 0; i < n; i++) v[i] = cJSON_DetachItemFromArray(arr, 0);
    for (int i = 1; i < n; i++) {
        cJSON *x = v[i]; double px = jget(x, "p"); int j = i - 1;
        while (j >= 0 && jget(v[j], "p") < px) { v[j + 1] = v[j]; j--; }
        v[j + 1] = x;
    }
    for (int i = 0; i < n; i++) cJSON_AddItemToArray(arr, v[i]);
    free(v);
}

/* utterance read-outs at row j (window-relative) */
static cJSON *readouts(const arctag_t *h, const run_t *r, int row, int spk_slot) {
    cJSON *ro = cJSON_CreateObject();
    if (!ro) return NULL;
    cJSON *intent = top(h, r, Q_UTT, row, "intent.", 3, NULL, NULL, NULL);
    append_all(intent, top(h, r, Q_UTT, row, "discourse.intent.", 3, NULL, NULL, NULL));
    sort_by_p_desc(intent);
    while (cJSON_GetArraySize(intent) > 4) cJSON_DeleteItemFromArray(intent, 4);
    cJSON_AddItemToObject(ro, "intent", intent);
    cJSON *act = top(h, r, Q_UTT, row, "discourse.act.", 3, NULL, NULL, NULL);
    append_all(act, top(h, r, Q_UTT, row, "discourse.answer", 1, NULL, NULL, NULL));
    append_all(act, top(h, r, Q_UTT, row, "discourse.backchannel", 1, NULL, NULL, NULL));
    append_all(act, top(h, r, Q_UTT, row, "discourse.assistant.", 1, NULL, NULL, NULL));
    cJSON_AddItemToObject(ro, "act", act);
    cJSON *saf = top(h, r, Q_UTT, row, "safety.", 20, NULL, NULL, NULL), *s2 = cJSON_CreateArray();
    for (cJSON *x = saf ? saf->child : NULL; x; x = x->next)
        if (cJSON_GetArraySize(s2) < 6 && jget(x, "p") >= 0.5 * jget(x, "thr")) cJSON_AddItemToArray(s2, cJSON_Duplicate(x, 1));
    cJSON_Delete(saf);
    cJSON_AddItemToObject(ro, "safety", s2);
    cJSON_AddItemToObject(ro, "emotion", top(h, r, Q_UTT, row, "affect.emotion.", 3, NULL, NULL, NULL));
    cJSON_AddItemToObject(ro, "arousal", top(h, r, Q_UTT, row, "affect.arousal.", 3, NULL, NULL, NULL));
    cJSON_AddItemToObject(ro, "role", spk_slot > 0 ? top(h, r, Q_SPK, row, "speaker.role.", 3, NULL, NULL, NULL) : cJSON_CreateArray());
    cJSON_AddItemToObject(ro, "item", top(h, r, Q_ITEM, row, "item.", 3, NULL, NULL, NULL));
    cJSON *st = cJSON_CreateObject();
    for (int g = 0; g < h->m.nsg; g++) {
        char pre[256]; snprintf(pre, sizeof pre, "%s.", h->m.state_groups[g]);
        cJSON *t = top(h, r, Q_FRAME, row, pre, 3, NULL, NULL, NULL);
        if (cJSON_GetArraySize(t)) cJSON_AddItemToObject(st, h->m.state_groups[g], t); else cJSON_Delete(t);
    }
    cJSON_AddItemToObject(ro, "state", st);
    return ro;
}

/* ------------------------------------------------------------------ spans */
typedef struct { int c, a, b; double p, thr; int head; int has_alt; int alt_c; double alt_p; int order; } span_t;

static const char *family_of(const char *t) {
    if (starts_with(t, "safety.pii.")) return "pii";
    if (starts_with(t, "safety.profanity.") || starts_with(t, "safety.toxicity.")) return "profanity";
    if (starts_with(t, "entity.number.") || starts_with(t, "entity.numeral")) return "number";
    if (starts_with(t, "entity.mention.")) return "entity";
    return "other";
}
static int decode_family(const char *t) {           /* -1 = none */
    if (!strcmp(t, "entity.numeral") || !strcmp(t, "safety.toxicity.toxic_any")) return -1;
    static const char *pre[5] = {"entity.number.", "entity.mention.", "safety.pii.", "safety.profanity.", "safety.toxicity."};
    for (int i = 0; i < 5; i++) if (starts_with(t, pre[i])) return i;
    return -1;
}

typedef struct { span_t *v; int n, cap; } spans_t;
static void sp_push(spans_t *s, span_t x) {
    if (s->n == s->cap) {
        int c = s->cap ? 2 * s->cap : 32; span_t *v = (span_t *)xrealloc(s->v, sizeof(span_t) * (size_t)c);
        if (!v) return;
        s->v = v; s->cap = c;
    }
    x.order = s->n; s->v[s->n++] = x;
}

static int cmp_xab(const void *x, const void *y) {      /* (a, b, order) */
    const span_t *p = (const span_t *)x, *q = (const span_t *)y;
    if (p->a != q->a) return p->a - q->a;
    if (p->b != q->b) return p->b - q->b;
    return p->order - q->order;
}
static int cmp_out(const void *x, const void *y) {      /* (a, -b, order) */
    const span_t *p = (const span_t *)x, *q = (const span_t *)y;
    if (p->a != q->a) return p->a - q->a;
    if (p->b != q->b) return q->b - p->b;
    return p->order - q->order;
}

/* decode spans over window rows [r0, r1) (global word index = row + off); keep spans with end > a0. ws[0] is word wb. */
static cJSON *decode_spans(arctag_t *h, const run_t *r, const unsigned char *us, int r0, int r1, int off, int a0, const word_t *ws, int wb) {
    const meta_t *m = &h->m; const profile_t *pf = &m->prof[h->prof];
    int nin = m->nc[Q_IN], W = r1 - r0;
    cJSON *out = cJSON_CreateArray();
    if (W <= 0) return out;
    unsigned char *ON = (unsigned char *)xcalloc((size_t)W * nin, 1), *BEG = (unsigned char *)xcalloc((size_t)W * nin, 1);
    unsigned char *headcol = (unsigned char *)xcalloc((size_t)nin, 1);
    if (!ON || !BEG || !headcol) { free(ON); free(BEG); free(headcol); return out; }
    for (int k = 0; k < W; k++) {
        int row = r0 + k;
        for (int c = 0; c < nin; c++) {
            ON[k * nin + c] = (double)P(h, r, Q_IN, row, c) >= (double)pf->thr[Q_IN][c] && pf->valid[Q_IN][c];
            int bo = m->beg_of_in[c];
            if (bo >= 0) BEG[k * nin + c] = (double)P(h, r, Q_BEG, row, bo) >= (double)pf->thr[Q_BEG][bo];
            if (us[row]) BEG[k * nin + c] = 1;
        }
    }
    /* span head families: their valid member columns */
    int nfc[MAX_FAM];
    for (int f = 0; f < m->nfam; f++) {
        nfc[f] = 0;
        if (!m->fam[f].use) continue;
        for (int t = 0; t < m->fam[f].ntags; t++) {
            int c = find_str(m->cols[Q_IN], nin, m->fam[f].tags[t]);
            if (c >= 0 && pf->valid[Q_IN][c]) { nfc[f]++; headcol[c] = 1; }
        }
    }
    spans_t res = {0};
    for (int c = 0; c < nin; c++) {
        if (headcol[c]) continue;
        int a = -1;
        for (int k = 0; k <= W; k++) {
            int on = k < W && ON[k * nin + c];
            if (a >= 0 && (!on || BEG[k * nin + c])) {
                int w0 = r0 + a + off, w1 = r0 + k - 1 + 1 + off;
                if (w1 > a0) {
                    double mx = 0; for (int t = a; t < k; t++) { double p = P(h, r, Q_IN, r0 + t, c); if (t == a || p > mx) mx = p; }
                    span_t s = {c, w0, w1, mx, pf->thr[Q_IN][c], 0, 0, 0, 0, 0}; sp_push(&res, s);
                }
                a = -1;
            }
            if (on && a < 0) a = k;
        }
    }
    /* type-exclusive merge per decode family (span_in path) */
    {
        spans_t keep = {0}, byf[5] = {{0}};
        for (int i = 0; i < res.n; i++) {
            int f = decode_family(m->cols[Q_IN][res.v[i].c]);
            if (f < 0) sp_push(&keep, res.v[i]); else sp_push(&byf[f], res.v[i]);
        }
        /* the python dict iterates families in first-seen order */
        int seen[5], ns = 0;
        for (int i = 0; i < res.n; i++) {
            int f = decode_family(m->cols[Q_IN][res.v[i].c]); if (f < 0) continue;
            int dup = 0; for (int j = 0; j < ns; j++) if (seen[j] == f) dup = 1;
            if (!dup) seen[ns++] = f;
        }
        spans_t outv = {0};
        for (int i = 0; i < keep.n; i++) sp_push(&outv, keep.v[i]);
        for (int si = 0; si < ns; si++) {
            spans_t *xs = &byf[seen[si]];
            qsort(xs->v, (size_t)xs->n, sizeof(span_t), cmp_xab);
            int cols[64], ncols = 0;
            for (int c = 0; c < nin && ncols < 64; c++) if (decode_family(m->cols[Q_IN][c]) == seen[si] && pf->valid[Q_IN][c]) cols[ncols++] = c;
            int i = 0;
            while (i < xs->n) {
                int ga = xs->v[i].a, gb = xs->v[i].b, j = i + 1;
                while (j < xs->n && xs->v[j].a < gb) { if (xs->v[j].b > gb) gb = xs->v[j].b; j++; }
                if (j - i == 1 || !ncols) { sp_push(&outv, xs->v[i]); i = j; continue; }
                double M[64]; int cnt = 0;
                for (int t = 0; t < ncols; t++) M[t] = 0;
                for (int w = ga; w < gb; w++) {
                    int row = w - off; if (row < r0 || row >= r1) continue;
                    cnt++; for (int t = 0; t < ncols; t++) M[t] += P(h, r, Q_IN, row, cols[t]);
                }
                for (int t = 0; t < ncols; t++) M[t] /= cnt > 0 ? cnt : 1;
                int b1 = 0; for (int t = 1; t < ncols; t++) if (M[t] > M[b1]) b1 = t;
                int b2 = -1; for (int t = 0; t < ncols; t++) if (t != b1 && (b2 < 0 || M[t] > M[b2])) b2 = t;
                double mx = 0; int first = 1;
                for (int w = ga; w < gb; w++) { int row = w - off; if (row < r0 || row >= r1) continue; double p = P(h, r, Q_IN, row, cols[b1]); if (first || p > mx) mx = p; first = 0; }
                span_t s = {cols[b1], ga, gb, mx, pf->thr[Q_IN][cols[b1]], 0, b2 >= 0, b2 >= 0 ? cols[b2] : 0, b2 >= 0 ? M[b2] : 0, 0};
                sp_push(&outv, s);
                i = j;
            }
        }
        free(keep.v); for (int f = 0; f < 5; f++) free(byf[f].v); free(res.v);
        res = outv;
    }
    /* span head (T0 greedy BIO) per family */
    for (int f = 0; f < m->nfam; f++) {
        if (!m->fam[f].use || !nfc[f]) continue;
        int it = out_index(h, m->fam[f].out_t), ib = out_index(h, m->fam[f].out_b);
        if (it < 0 || ib < 0) continue;
        int mem[64], nm = 0;
        for (int t = 0; t < m->fam[f].ntags && nm < 64; t++) {
            int c = find_str(m->cols[Q_IN], nin, m->fam[f].tags[t]);
            if (c >= 0 && headcol[c] && pf->valid[Q_IN][c]) mem[nm++] = t;
        }
        double th = pf->head_thr[f];
        int *lab = (int *)xcalloc((size_t)W, sizeof(int));
        if (!lab) continue;
        for (int k = 0; k < W; k++) {
            const float *pb = r->o[ib] + (size_t)(r0 + k) * r->st[ib];
            float pin = 1.0f - pb[0];                                /* float32 (= reference) */
            lab[k] = pin < (float)th ? 0 : ((us[r0 + k] || pb[1] > pb[2]) ? 1 : 2);
        }
        for (int k = 1; k < W; k++) if (lab[k] == 2 && lab[k - 1] == 0) lab[k] = 1;
        if (W) lab[0] = lab[0] ? 1 : 0;
        int k = 0;
        while (k < W) {
            if (!lab[k]) { k++; continue; }
            int e = k + 1;
            while (e < W && lab[e] == 2) e++;
            int w0 = r0 + k + off, w1 = r0 + e - 1 + 1 + off;
            if (w1 > a0) {
                double sc[64], pm[64], mx = 0;
                for (int t = 0; t < nm; t++) { sc[t] = 0; pm[t] = 0; }
                for (int x = k; x < e; x++) {
                    const float *pt = r->o[it] + (size_t)(r0 + x) * r->st[it];
                    for (int t = 0; t < nm; t++) {
                        float v = pt[mem[t]]; if (v < 1e-9f) v = 1e-9f; if (v > 1.0f) v = 1.0f;
                        sc[t] += logf(v); pm[t] += pt[mem[t]];
                    }
                    float pin = 1.0f - r->o[ib][(size_t)(r0 + x) * r->st[ib]];
                    if (x == k || pin > mx) mx = pin;
                }
                int b1 = 0; for (int t = 1; t < nm; t++) if (sc[t] > sc[b1]) b1 = t;
                int b2 = -1; for (int t = 0; t < nm; t++) if (t != b1 && (b2 < 0 || sc[t] > sc[b2])) b2 = t;
                int c = find_str(m->cols[Q_IN], nin, m->fam[f].tags[mem[b1]]);
                span_t s = {c, w0, w1, mx, th, 1, b2 >= 0, 0, 0, 0};
                if (b2 >= 0) { s.alt_c = find_str(m->cols[Q_IN], nin, m->fam[f].tags[mem[b2]]); s.alt_p = pm[b2] / (e - k); }
                sp_push(&res, s);
            }
            k = e;
        }
        free(lab);
    }
    /* python: hth[c] is set for every head span's column; a span_in-path span on such a column would show the head thr */
    for (int i = 0; i < res.n; i++) res.v[i].order = i;
    if (res.n > 1) qsort(res.v, (size_t)res.n, sizeof(span_t), cmp_out);
    int vm = out_index(h, "ver_m"), ve = out_index(h, "ver_e");
    for (int i = 0; i < res.n; i++) {
        span_t *s = &res.v[i];
        const char *tag = m->cols[Q_IN][s->c];
        cJSON *o = cJSON_CreateObject();
        cJSON_AddStringToObject(o, "type", tag);
        cJSON_AddStringToObject(o, "family", family_of(tag));
        cJSON_AddNumberToObject(o, "start", s->a);
        cJSON_AddNumberToObject(o, "end", s->b);
        char *txt = join_words(ws, wb, s->a, s->b); cJSON_AddStringToObject(o, "text", txt); free(txt);
        cJSON_AddItemToObject(o, "score", jnum(s->p));
        cJSON_AddItemToObject(o, "thr", jnum(s->thr));
        cJSON_AddStringToObject(o, "src", s->head ? "head" : "span_in");
        if (s->has_alt) {
            cJSON *al = cJSON_CreateObject();
            cJSON_AddStringToObject(al, "tag", m->cols[Q_IN][s->alt_c]); cJSON_AddItemToObject(al, "p", jnum(s->alt_p));
            cJSON_AddItemToObject(o, "alt", al);
        }
        if (m->is_norm[s->c] && vm >= 0 && ve >= 0) {
            double mean = 0; int cnt = 0;
            for (int w = s->a; w < s->b; w++) { int row = w - off; if (row >= 0 && row < r->W) { mean += r->o[vm][(size_t)row * r->st[vm]]; cnt++; } }
            int rl = s->b - 1 - off;
            double v = (cnt ? mean / cnt : 0) + (rl >= 0 && rl < r->W ? r->o[ve][(size_t)rl * r->st[ve]] : 0);
            cJSON_AddItemToObject(o, "ver", jnum(1.0 / (1.0 + exp(-v))));
        }
        cJSON_AddItemToArray(out, o);
    }
    free(res.v); free(ON); free(BEG); free(headcol);
    return out;
}

/* T0 events of rows [r0, r1); dedup: the same tag within the previous 3 words. ws[0] is word wb. */
static void decode_events(arctag_t *h, const run_t *r, int r0, int r1, int off, const word_t *ws, int wb, cJSON *out) {
    const profile_t *pf = &h->m.prof[h->prof];
    static const int qs[3] = {Q_EVAT, Q_EVIMM, Q_EVREC};
    for (int row = r0; row < r1; row++) {
        int i = row + off;
        for (int qi = 0; qi < 3; qi++) {
            int q = qs[qi];
            for (int c = 0; c < h->m.nc[q]; c++) {
                float p = P(h, r, q, row, c);
                if (!((double)p >= (double)pf->thr[q][c] && pf->valid[q][c])) continue;
                int dup = 0;
                for (int k = 0; k < h->nk && !dup; k++) {
                    const evkey_t *e = &h->keys[k];
                    /* the same tag name may appear in two read-outs: compare names */
                    if (e->i >= i - 3 && e->i <= i && !strcmp(h->m.cols[e->q][e->c], h->m.cols[q][c])) dup = 1;
                }
                if (dup) continue;
                if (h->nk == h->kcap) {
                    int kc = h->kcap ? 2 * h->kcap : 64; evkey_t *nk = (evkey_t *)xrealloc(h->keys, sizeof(evkey_t) * (size_t)kc);
                    if (!nk) continue;
                    h->keys = nk; h->kcap = kc;
                }
                h->keys[h->nk].q = q; h->keys[h->nk].c = c; h->keys[h->nk].i = i; h->nk++;
                cJSON *o = cJSON_CreateObject();
                cJSON_AddStringToObject(o, "type", h->m.cols[q][c]);
                cJSON_AddStringToObject(o, "q", QN[q]);
                cJSON_AddNumberToObject(o, "word", i);
                cJSON_AddItemToObject(o, "score", jnum(p));
                cJSON_AddItemToObject(o, "thr", jnum(pf->thr[q][c]));
                cJSON_AddStringToObject(o, "text", ws[i - wb].w);
                cJSON_AddItemToArray(out, o);
            }
        }
    }
}

/* ------------------------------------------------------------------ words */
static void word_free(word_t *w) { free(w->w); free(w->spk); memset(w, 0, sizeof *w); }

static void set_pieces(word_t *w, const int32_t *ids, int k, int32_t last) {
    if (k <= 0) { w->pcs[0] = 0; w->npc = 1; return; }
    if (k > P_MAX) { for (int i = 0; i < P_MAX - 1; i++) w->pcs[i] = ids[i]; w->pcs[P_MAX - 1] = last; w->npc = P_MAX; }
    else { for (int i = 0; i < k; i++) w->pcs[i] = ids[i]; w->npc = k; }
}

static int encode_str(arctag_t *h, const char *s, size_t n, int32_t *ids, int cap) {
    char *buf = (char *)xmalloc(n + 2);
    if (!buf) return -1;
    buf[0] = ' '; memcpy(buf + 1, s, n); buf[n + 1] = 0;
    int k = arctok_encode(h->tok, buf, ids, cap);
    free(buf);
    if (k < 0) tls_oom = 1;                                      /* the tokenizer's only failure */
    return k;
}

/* pieces: first 7 and last. Merging is quadratic in a word's length, so an absurdly long word is tokenized by its two ends. */
static void word_tokenize(arctag_t *h, word_t *w) {
    /* " " + strip(word) */
    const char *s = w->w; size_t n = strlen(s), a = 0, b = n;
    while (a < b) { size_t i = a; uint32_t cp; utf8_decode(s, n, &i, &cp); if (!uni_is_space(cp)) break; a = i; }
    while (b > a) {                                                          /* strip trailing whitespace */
        size_t st = b - 1; while (st > a && ((unsigned char)s[st] & 0xC0) == 0x80) st--;
        size_t i = st; uint32_t cp; utf8_decode(s, n, &i, &cp);
        if (!uni_is_space(cp)) break;
        b = st;
    }
    int32_t ids[256];
    if (b - a > MAX_WORD_BYTES) {
        int k = encode_str(h, s + a, MAX_WORD_BYTES / 2, ids, 256);
        size_t t = b - MAX_WORD_BYTES / 2; while (t < b && ((unsigned char)s[t] & 0xC0) == 0x80) t++;   /* whole characters */
        int32_t tail[256]; int kt = encode_str(h, s + t, b - t, tail, 256);
        int32_t last = kt > 0 ? tail[kt <= 256 ? kt - 1 : 255] : 0;
        if (kt > 256) { int32_t *big = (int32_t *)xmalloc(sizeof(int32_t) * (size_t)kt); if (big) { encode_str(h, s + t, b - t, big, kt); last = big[kt - 1]; free(big); } }
        set_pieces(w, ids, k > 256 ? 256 : k, last);
        if (w->npc < P_MAX && k > 0) { w->pcs[w->npc - 1] = last; }
        return;
    }
    int k = encode_str(h, s + a, b - a, ids, 256);
    if (k > 256) {                                                         /* very long word: need the last id too */
        int32_t *big = (int32_t *)xmalloc(sizeof(int32_t) * (size_t)k), last = ids[255];
        if (big) { encode_str(h, s + a, b - a, big, k); last = big[k - 1]; free(big); }
        set_pieces(w, ids, 256, last);
        return;
    }
    set_pieces(w, ids, k, k > 0 ? ids[k - 1] : 0);
}

/* a word's own facts, the same whichever path it arrives by */
static void word_finish(arctag_t *h, word_t *w) {
    if (!w->w) return;
    word_tokenize(h, w);
    digit_flags(w->w, &w->hd, &w->all);
}

static const char *first_str(cJSON *o, const char *const *keys, int n) {
    for (int i = 0; i < n; i++) { cJSON *x = cJSON_GetObjectItem(o, keys[i]); if (cJSON_IsString(x)) return x->valuestring; }
    return NULL;
}
static int first_num(cJSON *o, const char *const *keys, int n, double *v) {
    for (int i = 0; i < n; i++) { cJSON *x = cJSON_GetObjectItem(o, keys[i]); if (cJSON_IsNumber(x)) { *v = x->valuedouble; return 1; } }
    return 0;
}

/* parse one word (string or object); returns 0 on success; *eou set if "eou": true */
static int parse_word(arctag_t *h, cJSON *x, word_t *w, int *eou) {
    memset(w, 0, sizeof *w); w->final = 1; w->last_spk_i = -1; if (eou) *eou = 0;
    static const char *kw[] = {"w", "word", "text"}, *k0[] = {"t0", "start"}, *k1[] = {"t1", "end"}, *kc[] = {"conf", "confidence"};
    if (cJSON_IsString(x)) w->w = xstrdup(x->valuestring);
    else if (cJSON_IsObject(x)) {
        const char *s = first_str(x, kw, 3);
        if (!s) return -1;
        w->w = xstrdup(s);
        w->has_t0 = first_num(x, k0, 2, &w->t0); w->has_t1 = first_num(x, k1, 2, &w->t1); w->has_conf = first_num(x, kc, 2, &w->conf);
        cJSON *sp = cJSON_GetObjectItem(x, "spk"); if (!sp) sp = cJSON_GetObjectItem(x, "speaker");
        if (cJSON_IsString(sp)) { if (!(w->spk = xstrdup(sp->valuestring))) { word_free(w); return -1; } }
        else if (cJSON_IsNumber(sp)) { char b[32]; snprintf(b, sizeof b, "%d", sp->valueint); if (!(w->spk = xstrdup(b))) { word_free(w); return -1; } }
        cJSON *f = cJSON_GetObjectItem(x, "final"); if (cJSON_IsBool(f)) w->final = cJSON_IsTrue(f);
        cJSON *e = cJSON_GetObjectItem(x, "eou"); if (eou && cJSON_IsTrue(e)) *eou = 1;
    } else return -1;
    if (!w->w) { word_free(w); return -1; }
    word_finish(h, w);
    return 0;
}

/* ------------------------------------------------------------------ public API */
const char *arctag_version(void) { return ARCTAG_VERSION; }
void arctag_free_str(char *s) { free(s); }
const char *arctag_last_error(const arctag_t *h) { return h ? h->err : ""; }

static char *path_join(const char *d, const char *f) {
    size_t n = strlen(d) + strlen(f) + 2; char *p = (char *)xmalloc(n);
    if (p) snprintf(p, n, "%s%s%s", d, (*d && d[strlen(d) - 1] != '/' && d[strlen(d) - 1] != '\\') ? "/" : "", f);
    return p;
}

/* meta and weights come from different files: the decoder indexes model output by the meta's columns, so they must agree */
static int check_consistent(arctag_t *h) {
    const meta_t *m = &h->m;
    for (int q = 0; q < NQ; q++) if (m->nc[q] != atn_out_dim(h->nm, q)) {
        set_err(h, "arctag_meta.json has %d columns for %s, the weights have %d: they are from different checkpoints", m->nc[q], QN[q], atn_out_dim(h->nm, q));
        return -1;
    }
    for (int p = 0; p < m->nprof; p++) if (m->prof[p].dom < 0 || m->prof[p].dom >= atn_num_domains(h->nm)) {
        set_err(h, "arctag_meta.json: profile %s has domain %d, the weights have %d", m->prof[p].name, m->prof[p].dom, atn_num_domains(h->nm));
        return -1;
    }
    for (int i = 0; i < m->nc[Q_IN]; i++) if (m->beg_of_in[i] >= m->nc[Q_BEG]) { set_err(h, "arctag_meta.json: beg_of_in[%d] out of range", i); return -1; }
    for (int f = 0; f < m->nfam; f++) {
        int it = -1, ib = -1;
        for (int i = 0; i < h->nout; i++) { if (!strcmp(h->out_names[i], m->fam[f].out_t)) it = i; if (!strcmp(h->out_names[i], m->fam[f].out_b)) ib = i; }
        if (it < 0 || ib < 0) continue;
        if (m->fam[f].ntags > atn_out_dim(h->nm, h->nat[it]) || atn_out_dim(h->nm, h->nat[ib]) < 3) {
            set_err(h, "arctag_meta.json: span head family %s has %d tags, the weights have %d columns", m->fam[f].name, m->fam[f].ntags, atn_out_dim(h->nm, h->nat[it]));
            return -1;
        }
    }
    return 0;
}

arctag_t *arctag_create(const char *model_dir, char *err, size_t errlen) {
    api_enter();
    if (err && errlen) err[0] = 0;
    if (!model_dir) { if (err && errlen) snprintf(err, errlen, "no model directory"); return NULL; }
    arctag_t *h = (arctag_t *)xcalloc(1, sizeof(arctag_t));
    if (!h) { if (err && errlen) snprintf(err, errlen, "out of memory"); return NULL; }
    char *p = path_join(model_dir, "arctag_meta.json");
    if (!p || load_meta(h, p)) { free(p); goto fail; }
    free(p);
    p = path_join(model_dir, "tokenizer.json");
    if (!p) goto fail;
    h->tok = arctok_load(p, h->err, sizeof h->err); free(p);
    if (!h->tok) goto fail;
    h->prof = h->m.default_prof; h->asr = h->m.default_asr; h->max_ctx = h->m.max_ctx; h->stream_ctx = 256;
    h->sdirty = INT_MAX;
    for (int q = 0; q < NQ; q++) { h->qo[q] = h->nout; if (!(h->out_names[h->nout++] = xstrdup(QN[q]))) goto fail; }
    for (int f = 0; f < h->m.nfam; f++) {
        if (!(h->out_names[h->nout++] = xstrdup(h->m.fam[f].out_t))) goto fail;
        if (!(h->out_names[h->nout++] = xstrdup(h->m.fam[f].out_b))) goto fail;
    }
    if (!(h->out_names[h->nout++] = xstrdup("ver_m")) || !(h->out_names[h->nout++] = xstrdup("ver_e"))) goto fail;
    p = path_join(model_dir, "arctag.atw");
    if (!p) goto fail;
    FILE *fa = fopen(p, "rb");
    if (fa) fclose(fa);
    else {                                                       /* the common mistake gets its own message */
        char *onnx = path_join(model_dir, "arctag.onnx"); FILE *fo = onnx ? fopen(onnx, "rb") : NULL;
        free(onnx);
        if (fo) { fclose(fo); set_err(h, "%s has arctag.onnx but no arctag.atw: run tools/export_model.py on it", model_dir); free(p); goto fail; }
    }
    h->nm = atn_open(p, h->err, sizeof h->err);
    free(p);
    if (!h->nm) goto fail;
    h->total = atn_out_total(h->nm);
    for (int i = 0; i < h->nout; i++) {
        h->nat[i] = -1;
        for (int k = 0; k < ATN_NOUT; k++) if (!strcmp(h->out_names[i], atn_out_name(k))) h->nat[i] = k;
        if (h->nat[i] < 0) { set_err(h, "arctag_meta.json names output %s, which this runtime does not have", h->out_names[i]); goto fail; }
        int off = 0; for (int q = 0; q < h->nat[i]; q++) off += atn_out_dim(h->nm, q);
        h->ooff[i] = off;
    }
    if (check_consistent(h)) goto fail;
    if (tls_oom) { set_err(h, "out of memory"); goto fail; }
    h->err[0] = 0;
    return h;
fail:
    if (err && errlen) snprintf(err, errlen, "%s", h->err[0] ? h->err : tls_oom ? "out of memory" : "arctag_create failed");
    arctag_destroy(h);
    return NULL;
}

/* Forget words, rows and bookkeeping. The stream's model state is kept allocated and restarted, so a reset costs nothing. */
static void stream_clear(arctag_t *h) {
    for (int i = h->wbase; i < h->n; i++) word_free(&h->words[i - h->wbase]);
    for (int i = 0; i < NSLOT; i++) { free(h->slot[i]); h->slot[i] = NULL; }
    free(h->old_spk); h->old_spk = NULL;
    h->n = 0; h->wbase = 0; h->nb = 0; h->bfloor = 0; h->np = 0; h->nk = 0; h->lo = 0;
    h->sdone = 0; h->rbase = 0; h->sdirty = INT_MAX;
}

void arctag_reset(arctag_t *h) {
    if (!h) return;
    stream_clear(h);
    if (h->st) atn_set_domain(h->st, h->m.prof[h->prof].dom);
}

void arctag_destroy(arctag_t *h) {
    if (!h) return;
    stream_clear(h);
    free(h->words); free(h->bounds); free(h->pend); free(h->keys); free(h->rows);
    atn_free(h->one); atn_free(h->st); atn_close(h->nm);
    for (int i = 0; i < h->nout; i++) free(h->out_names[i]);
    free_meta(&h->m);
    arctok_free(h->tok);
    free(h);
}

int arctag_set_option(arctag_t *h, const char *key, const char *value) {
    if (!h || !key || !value) return -1;
    api_enter();
    if (!strcmp(key, "domain")) {
        int p = -1;
        for (int i = 0; i < h->m.nprof; i++) if (!strcmp(h->m.prof[i].name, value)) p = i;
        for (int i = 0; i < h->m.nalias && p < 0; i++) if (!strcmp(h->m.alias_k[i], value)) p = h->m.alias_v[i];
        if (p < 0) { set_err(h, "unknown domain '%s'", value); return -1; }
        h->prof = p; arctag_reset(h); return 0;
    }
    if (!strcmp(key, "asr")) {
        int a = find_str(h->m.asr, h->m.nasr, value);
        if (a < 0) { set_err(h, "unknown asr source '%s'", value); return -1; }
        h->asr = a; arctag_reset(h); return 0;
    }
    if (!strcmp(key, "threads")) return 0;                /* accepted and ignored: the runtime is single-threaded, see arctag.h */
    if (!strcmp(key, "max_ctx")) { int v = atoi(value); if (v < 16 || v > 2048) { set_err(h, "max_ctx must be 16..2048"); return -1; } h->max_ctx = v; return 0; }
    if (!strcmp(key, "stream_ctx")) {
        int v = atoi(value);
        if (v < 16 || v > 2048) { set_err(h, "stream_ctx must be 16..2048"); return -1; }
        if (v != h->stream_ctx) {                         /* the next poll replays the held words with the new window */
            h->stream_ctx = v; atn_free(h->st); h->st = NULL;
            h->sdone = h->wbase; h->rbase = h->wbase; h->sdirty = INT_MAX;
        }
        return 0;
    }
    set_err(h, "unknown option '%s'", key);
    return -1;
}

static char *finish_doc(arctag_t *h, cJSON *o) {
    char *s = o ? cJSON_PrintUnformatted(o) : NULL;
    cJSON_Delete(o);
    if (tls_oom || !s) { free(s); set_err(h, "out of memory"); return NULL; }
    return s;
}

static void merge_into(cJSON *o, cJSON *ro) {
    if (!o || !ro) { cJSON_Delete(ro); return; }
    for (cJSON *x = ro->child; x;) { cJSON *nx = x->next; const char *k = x->string; cJSON *it = cJSON_DetachItemViaPointer(ro, x); if (!cJSON_AddItemToObject(o, k, it)) cJSON_Delete(it); x = nx; }
    cJSON_Delete(ro);
}

/* offline tagging of words[0..n) with utterance starts us */
static char *tag_words_impl(arctag_t *h, word_t *ws, int n, unsigned char *us) {
    cJSON *res = cJSON_CreateObject();
    cJSON_AddStringToObject(res, "profile", h->m.prof[h->prof].name);
    cJSON_AddNumberToObject(res, "n_words", n);
    cJSON *wa = cJSON_AddArrayToObject(res, "words");
    for (int i = 0; i < n; i++) cJSON_AddItemToArray(wa, cJSON_CreateString(ws[i].w));
    cJSON *spans = cJSON_AddArrayToObject(res, "spans"), *events = cJSON_AddArrayToObject(res, "events"), *utts = cJSON_AddArrayToObject(res, "utterances");
    int saved_nk = h->nk; evkey_t *saved_keys = h->keys; int saved_cap = h->kcap;
    h->nk = 0; h->keys = NULL; h->kcap = 0;
    int ok = 1;
    for (int c0 = 0; c0 < n && ok; c0 += h->max_ctx) {
        int c1 = c0 + h->max_ctx < n ? c0 + h->max_ctx : n, W = c1 - c0;
        unsigned char *u = (unsigned char *)xmalloc((size_t)W);
        if (!u) { ok = 0; break; }
        memcpy(u, us + c0, (size_t)W); u[0] = 1;
        /* features restart at every chunk (= python reference): the chunk is its own word list */
        run_t r;
        if (run_model(h, ws + c0, W, u, &r)) { free(u); ok = 0; break; }
        append_all(spans, decode_spans(h, &r, u, 0, W, c0, -1, ws, 0));
        decode_events(h, &r, 0, W, c0, ws, 0, events);
        int a = 0;
        for (int i = 1; i <= W; i++) {
            if (i < W && !u[i]) continue;
            cJSON *o = cJSON_CreateObject();
            cJSON_AddNumberToObject(o, "start", a + c0); cJSON_AddNumberToObject(o, "end", i + c0);
            char *t = join_words(ws, 0, a + c0, i + c0); cJSON_AddStringToObject(o, "text", t); free(t);
            merge_into(o, readouts(h, &r, i - 1, ws[i + c0 - 1].sp));
            if (!cJSON_AddItemToArray(utts, o)) cJSON_Delete(o);
            a = i;
        }
        run_free(&r); free(u);
    }
    free(h->keys); h->keys = saved_keys; h->nk = saved_nk; h->kcap = saved_cap;
    if (!ok) { cJSON_Delete(res); if (tls_oom) set_err(h, "out of memory"); return NULL; }
    return finish_doc(h, res);
}

static int is_sent_end(const char *w) {
    size_t n = strlen(w);
    while (n && strchr("\"')]}", w[n - 1])) n--;
    return n && strchr(".?!", w[n - 1]) != NULL;
}

char *arctag_tag_text(arctag_t *h, const char *text) {
    if (!h || !text) return NULL;
    api_enter();
    int cap = 64, n = 0;
    word_t *ws = (word_t *)xcalloc((size_t)cap, sizeof(word_t));
    unsigned char *us = (unsigned char *)xcalloc((size_t)cap, 1);
    char *out = NULL;
    if (!ws || !us) goto done;
    const char *s = text; int nxt = 1;
    while (*s) {
        while (*s && strchr(" \t\n\r\f\v", *s)) s++;
        if (!*s) break;
        const char *e = s; while (*e && !strchr(" \t\n\r\f\v", *e)) e++;
        if (n == cap) {
            word_t *w2 = (word_t *)xrealloc(ws, sizeof(word_t) * (size_t)cap * 2); if (w2) ws = w2;
            unsigned char *u2 = (unsigned char *)xrealloc(us, (size_t)cap * 2); if (u2) us = u2;
            if (!w2 || !u2) goto done;
            cap *= 2;
        }
        memset(&ws[n], 0, sizeof(word_t));
        ws[n].w = (char *)xmalloc((size_t)(e - s) + 1);
        if (!ws[n].w) goto done;
        memcpy(ws[n].w, s, (size_t)(e - s)); ws[n].w[e - s] = 0; ws[n].final = 1;
        word_finish(h, &ws[n]);
        us[n] = (unsigned char)nxt; nxt = is_sent_end(ws[n].w);
        n++; s = e;
    }
    out = tag_words_impl(h, ws, n, us);
done:
    if (ws) for (int i = 0; i < n; i++) word_free(&ws[i]);
    free(ws); free(us);
    if (tls_oom && !out) set_err(h, "out of memory");
    return out;
}

char *arctag_tag_words(arctag_t *h, const char *words_json) {
    if (!h || !words_json) return NULL;
    api_enter();
    cJSON *j = cJSON_Parse(words_json);
    if (!cJSON_IsArray(j)) { set_err(h, "tag_words: expected a JSON array"); cJSON_Delete(j); return NULL; }
    int n = cJSON_GetArraySize(j), k = 0, nxt = 1;
    word_t *ws = (word_t *)xcalloc((size_t)n + 1, sizeof(word_t));
    unsigned char *us = (unsigned char *)xcalloc((size_t)n + 1, 1);
    char *out = NULL;
    if (ws && us) {
        for (cJSON *x = j->child; x; x = x->next) {
            int eou;
            if (parse_word(h, x, &ws[k], &eou)) { set_err(h, "tag_words: bad word at %d", k); continue; }
            us[k] = (unsigned char)nxt; nxt = eou; k++;
        }
        out = tag_words_impl(h, ws, k, us);
        for (int i = 0; i < k; i++) word_free(&ws[i]);
    }
    cJSON_Delete(j);
    free(ws); free(us);
    if (tls_oom && !out) set_err(h, "out of memory");
    return out;
}

/* ------------------------------------------------------------------ the stream */
static word_t *word_at(const arctag_t *h, int i) { return &h->words[i - h->wbase]; }

static int add_bound(arctag_t *h) {
    int last = h->nb ? h->bounds[h->nb - 1] : h->bfloor;
    if (!h->n || last >= h->n) return 0;
    if (h->nb == h->bcap) {
        int c = h->bcap ? 2 * h->bcap : 64; int *v = (int *)xrealloc(h->bounds, sizeof(int) * (size_t)c);
        if (!v) return -1;
        h->bounds = v; h->bcap = c;
    }
    if (h->np == h->pcap) {
        int c = h->pcap ? 2 * h->pcap : 64; uspan_t *v = (uspan_t *)xrealloc(h->pend, sizeof(uspan_t) * (size_t)c);
        if (!v) return -1;
        h->pend = v; h->pcap = c;
    }
    h->pend[h->np].start = last; h->pend[h->np].end = h->n; h->np++;
    h->bounds[h->nb++] = h->n;
    return 0;
}

/* The oldest revision possible: word from - 1's feature state must still be held. */
static int revision_floor(const arctag_t *h) { return h->wbase ? h->wbase + 1 : 0; }

/* Append a word whose text and timing are parsed: its features follow from the word before it. */
static int stream_append(arctag_t *h, word_t *w) {
    if (h->n - h->wbase == h->cap) {
        int c = h->cap ? 2 * h->cap : 256; word_t *v = (word_t *)xrealloc(h->words, sizeof(word_t) * (size_t)c);
        if (!v) return -1;
        h->words = v; h->cap = c;
    }
    const word_t *prev = h->n > h->wbase ? word_at(h, h->n - 1) : NULL;
    const char *prev_spk = NULL;
    if (prev && prev->last_spk_i >= 0) prev_spk = prev->last_spk_i >= h->wbase ? word_at(h, prev->last_spk_i)->spk : h->old_spk;
    if (word_feat(h, w, prev, h->n, prev_spk, h->slot, 1)) return -1;
    *word_at(h, h->n) = *w;
    h->n++;
    return 0;
}

int arctag_push_words(arctag_t *h, const char *words_json) {
    if (!h || !words_json) return -1;
    api_enter();
    cJSON *j = cJSON_Parse(words_json);
    if (!j) { set_err(h, tls_oom ? "out of memory" : "push_words: invalid JSON"); return -1; }
    cJSON *arr = NULL; int from = h->n, endpoint = 0, rc = -1, nw = 0;
    word_t *fresh = NULL; unsigned char *eous = NULL;
    if (cJSON_IsArray(j)) arr = j;
    else if (cJSON_IsObject(j)) {
        arr = cJSON_GetObjectItem(j, "words");
        if (arr && !cJSON_IsArray(arr)) { set_err(h, "push_words: \"words\" must be an array"); goto done; }
        cJSON *f = cJSON_GetObjectItem(j, "from"); if (cJSON_IsNumber(f)) from = f->valueint;
        endpoint = cJSON_IsTrue(cJSON_GetObjectItem(j, "endpoint"));
    } else { set_err(h, "push_words: expected an array or an object"); goto done; }
    if (from < 0) from = 0;
    if (from > h->n) from = h->n;
    int count = arr ? cJSON_GetArraySize(arr) : 0, revising = arr && from < h->n;
    if (revising && from < revision_floor(h)) {
        set_err(h, "push_words: cannot revise from word %d: words before %d are no longer held (see \"Long sessions\" in the README)", from, revision_floor(h));
        goto done;
    }
    {
        int after = (arr ? from : h->n) + count, done_to = revising && from < h->sdone ? from : h->sdone;
        if (count > MAX_BACKLOG || after - done_to > MAX_BACKLOG) { set_err(h, "push_words: more than %d words waiting for arctag_poll", MAX_BACKLOG); goto done; }
    }
    /* parse everything first: a bad word leaves the stream untouched */
    if (count) {
        fresh = (word_t *)xcalloc((size_t)count, sizeof(word_t)); eous = (unsigned char *)xcalloc((size_t)count, 1);
        if (!fresh || !eous) goto done;
    }
    for (cJSON *x = arr ? arr->child : NULL; x; x = x->next) {
        int eou;
        if (parse_word(h, x, &fresh[nw], &eou)) { set_err(h, tls_oom ? "out of memory" : "push_words: bad word at %d", nw); goto done; }
        eous[nw++] = (unsigned char)eou;
    }
    if (tls_oom) { set_err(h, "out of memory"); goto done; }
    if (revising) {                                                 /* drop words[from:] and later boundaries */
        for (int i = from; i < h->n; i++) word_free(word_at(h, i));
        h->n = from;
        if (from < h->sdirty) h->sdirty = from;
        while (h->nb && h->bounds[h->nb - 1] > from) h->nb--;
        int k = 0; for (int i = 0; i < h->np; i++) if (h->pend[i].end <= from) h->pend[k++] = h->pend[i];
        h->np = k;
        int keep = from > h->wbase ? word_at(h, from - 1)->nslot : 0;
        for (int i = keep; i < NSLOT; i++) { free(h->slot[i]); h->slot[i] = NULL; }
    }
    if (from < h->lo) h->lo = from;
    rc = 0;
    int put = 0;
    for (; put < nw; put++) {
        if (stream_append(h, &fresh[put])) { set_err(h, "out of memory"); rc = -1; break; }
        if (eous[put] && add_bound(h)) { put++; set_err(h, "out of memory"); rc = -1; break; }
    }
    if (rc == 0 && endpoint && add_bound(h)) { set_err(h, "out of memory"); rc = -1; }
    if (rc == 0 && tls_oom) { set_err(h, "out of memory"); rc = -1; }
    /* the rest, on failure, are not part of the stream */
    for (int i = put; i < nw; i++) word_free(&fresh[i]);
    nw = 0;
done:
    for (int i = 0; i < nw; i++) word_free(&fresh[i]);
    free(fresh); free(eous);
    cJSON_Delete(j);
    return rc;
}

/* words, rows: how much of the session is held. Rows: the newest decode window plus what a revision may reach. Words: those rows
 * plus an attention window, so a revision too deep for the snapshots can still rebuild a full window by replay. */
static int rows_keep(const arctag_t *h) { return h->m.span_win + 16 + REWIND_ROWS; }
static int words_keep(const arctag_t *h) { return rows_keep(h) + h->stream_ctx + CHUNK; }

/* Move the windows forward; amortised constant work per word. Nothing here is visible in the output. */
static void stream_compact(arctag_t *h) {
    /* the words whose text a pending utterance or an undecoded row needs were all reported before this is called */
    int to = h->sdone - words_keep(h);
    if (to - h->wbase >= CHUNK) {
        for (int i = h->wbase; i < to; i++) {
            word_t *w = word_at(h, i);
            if (w->spk) { free(h->old_spk); h->old_spk = w->spk; w->spk = NULL; }
            word_free(w);
        }
        memmove(h->words, h->words + (to - h->wbase), sizeof(word_t) * (size_t)(h->n - to));
        h->wbase = to;
        int k = 0;
        while (k < h->nb && h->bounds[k] < to) h->bfloor = h->bounds[k++];
        if (k) { memmove(h->bounds, h->bounds + k, sizeof(int) * (size_t)(h->nb - k)); h->nb -= k; }
    }
    int kk = 0;
    for (int i = 0; i < h->nk; i++) if (h->keys[i].i >= h->rbase - 4) h->keys[kk++] = h->keys[i];
    h->nk = kk;
}

static int is_bound(const arctag_t *h, int i) {
    int lo = 0, hi = h->nb - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (h->bounds[mid] == i) return 1;
        if (h->bounds[mid] < i) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}

/* make room for c more rows: drop the oldest beyond what decoding and revisions can use */
static void rows_room(arctag_t *h, int c) {
    int held = h->sdone - h->rbase, keep = rows_keep(h);
    if (held + c <= h->rcap) return;
    int drop = held - keep;
    if (drop <= 0) return;
    memmove(h->rows, h->rows + (size_t)drop * h->total, sizeof(float) * (size_t)(held - drop) * (size_t)h->total);
    h->rbase += drop;
}

/* events and closed utterances of the rows not reported yet */
static void report_rows(arctag_t *h, cJSON *ev, cJSON *utts) {
    run_t r; memset(&r, 0, sizeof r);
    int W = h->sdone - h->rbase;
    run_view(h, &r, h->rows, W);
    int lo = h->lo > h->rbase ? h->lo : h->rbase;
    if (lo < h->sdone) decode_events(h, &r, lo - h->rbase, W, h->rbase, h->words, h->wbase, ev);
    if (h->lo < h->sdone) h->lo = h->sdone;
    int k = 0;
    for (int i = 0; i < h->np; i++) {
        int start = h->pend[i].start, end = h->pend[i].end;
        if (end > h->sdone) { h->pend[k++] = h->pend[i]; continue; }
        if (end - 1 < h->rbase) continue;                            /* cannot happen in normal use: rows outlive reporting */
        cJSON *o = cJSON_CreateObject();
        cJSON_AddNumberToObject(o, "start", start); cJSON_AddNumberToObject(o, "end", end);
        /* text: only held words. An utterance longer than the held words loses its head, see the README. */
        char *t = join_words(h->words, h->wbase, start > h->wbase ? start : h->wbase, end); cJSON_AddStringToObject(o, "text", t); free(t);
        cJSON_AddBoolToObject(o, "final", 1);
        merge_into(o, readouts(h, &r, end - 1 - h->rbase, word_at(h, end - 1)->sp));
        if (!cJSON_AddItemToArray(utts, o)) cJSON_Delete(o);
    }
    h->np = k;
}

/* step every word not stepped yet, a chunk at a time, so a large backlog costs no more row memory than a small one */
static int stream_advance(arctag_t *h, cJSON *ev, cJSON *utts) {
    if (!h->st) {
        h->st = atn_new(h->nm, h->m.prof[h->prof].dom, h->stream_ctx + ATN_REWIND_WORDS, h->stream_ctx);
        if (!h->st) { set_err(h, "out of memory"); tls_oom = 1; return -1; }
        atn_restart(h->st, h->sdone);
    }
    if (!h->rows) {
        h->rcap = rows_keep(h) + CHUNK;
        h->rows = (float *)xmalloc(sizeof(float) * (size_t)h->rcap * (size_t)h->total);
        if (!h->rows) { set_err(h, "out of memory"); return -1; }
    }
    if (h->sdirty < h->sdone) {
        int p = atn_rewind(h->st, h->sdirty);
        if (p < 0) {                                                 /* deeper than the snapshots: replay what is held from a fresh state */
            p = atn_first(h->st) > h->wbase ? atn_first(h->st) : h->wbase;
            atn_restart(h->st, p);
        }
        h->sdone = p;
        if (p < h->rbase) h->rbase = p;
    }
    h->sdirty = INT_MAX;
    atn_word in[CHUNK];
    while (h->sdone < h->n) {
        int c = h->n - h->sdone < CHUNK ? h->n - h->sdone : CHUNK;
        rows_room(h, c);
        for (int k = 0; k < c; k++) {
            int i = h->sdone + k; const word_t *w = word_at(h, i);
            word_input(h, w, w->spk_any ? 2 : 0, i == 0 || is_bound(h, i), &in[k]);
        }
        atn_steps(h->st, in, c, h->rows + (size_t)(h->sdone - h->rbase) * h->total);
        h->sdone += c;
        report_rows(h, ev, utts);
        stream_compact(h);
    }
    report_rows(h, ev, utts);                                        /* an endpoint with no new words */
    return 0;
}

char *arctag_poll(arctag_t *h) {
    if (!h) return NULL;
    api_enter();
    int n = h->n;
    cJSON *res = cJSON_CreateObject();
    cJSON_AddNumberToObject(res, "n_words", n);
    int a0 = n - h->m.span_win > 0 ? n - h->m.span_win : 0;
    cJSON_AddNumberToObject(res, "a0", a0);
    if (!n) {
        cJSON_AddArrayToObject(res, "spans"); cJSON_AddArrayToObject(res, "events"); cJSON_AddArrayToObject(res, "utterances");
        cJSON_AddNullToObject(res, "current");
        return finish_doc(h, res);
    }
    cJSON *ev = cJSON_CreateArray(), *utts = cJSON_CreateArray(), *spans = NULL;
    unsigned char *us = NULL;
    if (stream_advance(h, ev, utts)) goto fail;
    int s = h->rbase, W = n - s;
    if (W < 1) { set_err(h, "poll: internal error, no rows"); goto fail; }
    us = (unsigned char *)xcalloc((size_t)W, 1);
    if (!us) goto fail;
    us[0] = 1;
    for (int b = 0; b < h->nb; b++) if (h->bounds[b] >= s && h->bounds[b] < n) us[h->bounds[b] - s] = 1;
    run_t r; memset(&r, 0, sizeof r);
    run_view(h, &r, h->rows, W);
    int d0 = a0 - 16 > s ? a0 - 16 : s;
    spans = decode_spans(h, &r, us, d0 - s, W, s, a0, h->words, h->wbase);
    if (!cJSON_AddItemToObject(res, "spans", spans)) { cJSON_Delete(spans); tls_oom = 1; }
    if (!cJSON_AddItemToObject(res, "events", ev)) { cJSON_Delete(ev); tls_oom = 1; }
    ev = NULL;
    if (!cJSON_AddItemToObject(res, "utterances", utts)) { cJSON_Delete(utts); tls_oom = 1; }
    utts = NULL;
    int ustart = h->nb ? h->bounds[h->nb - 1] : h->bfloor;
    if (ustart < n) {
        cJSON *o = cJSON_CreateObject();
        cJSON_AddNumberToObject(o, "start", ustart); cJSON_AddNumberToObject(o, "end", n);
        cJSON_AddBoolToObject(o, "final", 0);
        merge_into(o, readouts(h, &r, W - 1, word_at(h, n - 1)->sp));
        if (!cJSON_AddItemToObject(res, "current", o)) { cJSON_Delete(o); tls_oom = 1; }
    } else cJSON_AddNullToObject(res, "current");
    free(us);
    return finish_doc(h, res);
fail:
    cJSON_Delete(ev); cJSON_Delete(utts); cJSON_Delete(res); free(us);
    if (tls_oom) set_err(h, "out of memory");
    return NULL;
}
