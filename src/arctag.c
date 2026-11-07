/* arctag: Arc Tagger C library (ONNX Runtime C API). Decoding = the Arc Tagger package (see README). */
#include "arctag.h"
#include "tokenizer.h"
#include "cJSON.h"
#include <onnxruntime_c_api.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define ARCTAG_VERSION "0.1.0"
#define NQ 10
#define P_MAX 8
#define NFEAT 8
#define MAX_FAM 8
#define MAX_OUT 32

static const char *QN[NQ] = {"frame", "span_beg", "span_in", "span_end", "ev_at", "ev_imm", "ev_rec", "utt", "speaker", "item"};
enum { Q_FRAME, Q_BEG, Q_IN, Q_END, Q_EVAT, Q_EVIMM, Q_EVREC, Q_UTT, Q_SPK, Q_ITEM };

/* ------------------------------------------------------------------ small utils */
static char *xstrdup(const char *s) { size_t n = strlen(s) + 1; char *c = (char *)malloc(n); if (c) memcpy(c, s, n); return c; }
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
    int npc; int64_t pcs[P_MAX];
} word_t;

typedef struct { int q, c, i; } evkey_t;

struct arctag {
    const OrtApi *ort; OrtEnv *env; OrtSession *sess; OrtMemoryInfo *mem;
    char *model_path; int threads;
    meta_t m; arctok_t *tok;
    int prof, asr, max_ctx, stream_ctx;
    char err[512];
    /* stream */
    word_t *words; int n, cap;
    int *bounds; int nb, bcap;            /* utterance end indices (exclusive), increasing */
    int *pend; int np, pcap;              /* closed utterances not yet reported (indices into bounds) */
    evkey_t *keys; int nk, kcap;
    int lo;                               /* lowest word index whose events are not yet checked */
    /* output names */
    int nout; char *out_names[MAX_OUT];
    int qo[NQ];                           /* output index of each read-out */
};

static void set_err(arctag_t *h, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); vsnprintf(h->err, sizeof h->err, fmt, ap); va_end(ap);
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); b = NULL; }
    fclose(f); if (b) b[n] = 0;
    return b;
}

static char **str_array(cJSON *a, int *n) {
    *n = cJSON_GetArraySize(a);
    char **o = (char **)calloc((size_t)(*n > 0 ? *n : 1), sizeof(char *));
    int i = 0;
    for (cJSON *x = a ? a->child : NULL; x; x = x->next, i++) o[i] = xstrdup(cJSON_IsString(x) ? x->valuestring : "");
    return o;
}

static int num_array(cJSON *a, double *o, int cap) {
    int i = 0;
    for (cJSON *x = a ? a->child : NULL; x && i < cap; x = x->next) o[i++] = x->valuedouble;
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
    cJSON *ro = cJSON_GetObjectItem(j, "readouts");
    for (int q = 0; q < NQ; q++) {
        cJSON *a = cJSON_GetObjectItem(ro, QN[q]);
        if (!cJSON_IsArray(a)) { set_err(h, "meta: readouts.%s missing", QN[q]); cJSON_Delete(j); return -1; }
        m->cols[q] = str_array(a, &m->nc[q]);
    }
    int nin = m->nc[Q_IN];
    m->beg_of_in = (int *)calloc((size_t)nin, sizeof(int)); m->is_norm = (int *)calloc((size_t)nin, sizeof(int));
    double tmp[1024];
    int k = num_array(cJSON_GetObjectItem(j, "beg_of_in"), tmp, 1024);
    for (int i = 0; i < nin; i++) m->beg_of_in[i] = i < k ? (int)tmp[i] : -1;
    k = num_array(cJSON_GetObjectItem(j, "is_norm"), tmp, 1024);
    for (int i = 0; i < nin; i++) m->is_norm[i] = i < k ? (int)tmp[i] : 0;
    cJSON *sh = cJSON_GetObjectItem(j, "span_head");
    for (cJSON *f = sh ? sh->child : NULL; f && m->nfam < MAX_FAM; f = f->next) {
        fam_t *F = &m->fam[m->nfam++];
        F->name = xstrdup(cJSON_GetObjectItem(f, "name")->valuestring);
        F->out_t = xstrdup(cJSON_GetObjectItem(f, "out_t")->valuestring);
        F->out_b = xstrdup(cJSON_GetObjectItem(f, "out_b")->valuestring);
        F->tags = str_array(cJSON_GetObjectItem(f, "tags"), &F->ntags);
        F->use = cJSON_IsTrue(cJSON_GetObjectItem(f, "use"));
    }
    m->state_groups = str_array(cJSON_GetObjectItem(j, "state_groups"), &m->nsg);
    m->n_gap = num_array(cJSON_GetObjectItem(j, "gap_edges"), m->gap_e, 16);
    m->n_dur = num_array(cJSON_GetObjectItem(j, "dur_edges"), m->dur_e, 16);
    m->n_conf = num_array(cJSON_GetObjectItem(j, "conf_edges"), m->conf_e, 16);
    m->asr = str_array(cJSON_GetObjectItem(j, "asr_sources"), &m->nasr);
    cJSON *da = cJSON_GetObjectItem(j, "default_asr");
    m->default_asr = cJSON_IsString(da) ? find_str(m->asr, m->nasr, da->valuestring) : 0;
    if (m->default_asr < 0) m->default_asr = 0;
    cJSON *x = cJSON_GetObjectItem(j, "max_ctx"); m->max_ctx = cJSON_IsNumber(x) ? x->valueint : 2048;
    x = cJSON_GetObjectItem(j, "span_win"); m->span_win = cJSON_IsNumber(x) ? x->valueint : 96;
    cJSON *pr = cJSON_GetObjectItem(j, "profiles");
    m->nprof = cJSON_GetArraySize(pr);
    m->prof = (profile_t *)calloc((size_t)m->nprof, sizeof(profile_t));
    int pi = 0;
    for (cJSON *p = pr ? pr->child : NULL; p; p = p->next, pi++) {
        profile_t *P = &m->prof[pi];
        P->name = xstrdup(p->string);
        P->dom = cJSON_GetObjectItem(p, "dom")->valueint;
        cJSON *th = cJSON_GetObjectItem(p, "thr"), *va = cJSON_GetObjectItem(p, "valid"), *ht = cJSON_GetObjectItem(p, "head_thr");
        for (int q = 0; q < NQ; q++) {
            P->thr[q] = (double *)calloc((size_t)m->nc[q] + 1, sizeof(double));
            P->valid[q] = (unsigned char *)calloc((size_t)m->nc[q] + 1, 1);
            cJSON *a = cJSON_GetObjectItem(th, QN[q]), *b = cJSON_GetObjectItem(va, QN[q]);
            int c = 0;
            for (cJSON *e = a ? a->child : NULL; e && c < m->nc[q]; e = e->next) P->thr[q][c++] = e->valuedouble;
            c = 0;
            for (cJSON *e = b ? b->child : NULL; e && c < m->nc[q]; e = e->next) P->valid[q][c++] = (unsigned char)(e->valuedouble != 0);
        }
        for (int f = 0; f < m->nfam; f++) {
            cJSON *v = cJSON_GetObjectItem(ht, m->fam[f].name);
            P->head_thr[f] = cJSON_IsNumber(v) ? v->valuedouble : 0.5;
        }
    }
    cJSON *al = cJSON_GetObjectItem(j, "aliases");
    m->nalias = cJSON_GetArraySize(al);
    m->alias_k = (char **)calloc((size_t)m->nalias + 1, sizeof(char *)); m->alias_v = (int *)calloc((size_t)m->nalias + 1, sizeof(int));
    int ai = 0;
    for (cJSON *a = al ? al->child : NULL; a; a = a->next, ai++) {
        m->alias_k[ai] = xstrdup(a->string);
        m->alias_v[ai] = 0;
        for (int p = 0; p < m->nprof; p++) if (cJSON_IsString(a) && !strcmp(m->prof[p].name, a->valuestring)) m->alias_v[ai] = p;
    }
    cJSON *dp = cJSON_GetObjectItem(j, "default_profile");
    m->default_prof = 0;
    for (int p = 0; p < m->nprof; p++) if (cJSON_IsString(dp) && !strcmp(m->prof[p].name, dp->valuestring)) m->default_prof = p;
    cJSON_Delete(j);
    return 0;
}

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
    for (int p = 0; p < m->nprof; p++) {
        free(m->prof[p].name);
        for (int q = 0; q < NQ; q++) { free(m->prof[p].thr[q]); free(m->prof[p].valid[q]); }
    }
    free(m->prof);
    for (int i = 0; i < m->nalias; i++) free(m->alias_k[i]);
    free(m->alias_k); free(m->alias_v);
}

/* ------------------------------------------------------------------ ONNX Runtime */
#define ORT_CHECK(h, expr) do { OrtStatus *st_ = (expr); if (st_) { set_err(h, "onnxruntime: %s", (h)->ort->GetErrorMessage(st_)); (h)->ort->ReleaseStatus(st_); goto fail; } } while (0)

static int open_session(arctag_t *h) {
    OrtSessionOptions *so = NULL;
    if (h->sess) { h->ort->ReleaseSession(h->sess); h->sess = NULL; }
    ORT_CHECK(h, h->ort->CreateSessionOptions(&so));
    ORT_CHECK(h, h->ort->SetIntraOpNumThreads(so, h->threads));
    ORT_CHECK(h, h->ort->SetInterOpNumThreads(so, 1));
    ORT_CHECK(h, h->ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
#ifdef _WIN32
    {
        int n = MultiByteToWideChar(CP_UTF8, 0, h->model_path, -1, NULL, 0);
        wchar_t *wp = (wchar_t *)malloc(sizeof(wchar_t) * (size_t)n);
        MultiByteToWideChar(CP_UTF8, 0, h->model_path, -1, wp, n);
        OrtStatus *st = h->ort->CreateSession(h->env, wp, so, &h->sess);
        free(wp);
        ORT_CHECK(h, st);
    }
#else
    ORT_CHECK(h, h->ort->CreateSession(h->env, h->model_path, so, &h->sess));
#endif
    h->ort->ReleaseSessionOptions(so);
    return 0;
fail:
    if (so) h->ort->ReleaseSessionOptions(so);
    return -1;
}

typedef struct {
    int W; OrtValue *vals[MAX_OUT]; float *o[MAX_OUT]; int dim[MAX_OUT];
} run_t;

static void run_free(arctag_t *h, run_t *r) {
    for (int i = 0; i < h->nout; i++) if (r->vals[i]) h->ort->ReleaseValue(r->vals[i]);
    memset(r, 0, sizeof *r);
}

static int out_index(const arctag_t *h, const char *name) {
    for (int i = 0; i < h->nout; i++) if (!strcmp(h->out_names[i], name)) return i;
    return -1;
}

static int bk(const double *e, int n, double x) { int k = 0; for (int i = 0; i < n; i++) if (e[i] <= x) k++; return 1 + k; }
static int clampi(int x, int a, int b) { return x < a ? a : x > b ? b : x; }

static int word_ncp(const char *s) { int n = 0; for (; *s; s++) if (((unsigned char)*s & 0xC0) != 0x80) n++; return n; }

/* word flags: has a digit (Python str.isdigit per char), all chars digits */
static void digit_flags(const char *s, int *hd, int *all) {
    size_t n = strlen(s), i = 0; int any = 0, every = n > 0;
    while (i < n) { uint32_t cp; utf8_decode(s, n, &i, &cp); if (uni_is_digit(cp)) any = 1; else every = 0; }
    *hd = any; *all = any && every;
}

/* model inputs for words[a..b) (features computed sequentially from word 0 like the engine), us = utterance starts */
static int run_model(arctag_t *h, const word_t *ws, int n_all, int a, int b, const unsigned char *us, run_t *r) {
    int W = b - a;
    memset(r, 0, sizeof *r); r->W = W;
    int64_t *pcs = (int64_t *)calloc((size_t)W * P_MAX, 8), *npc = (int64_t *)calloc((size_t)W, 8), *wfl = (int64_t *)calloc((size_t)W * 5, 8),
            *feats = (int64_t *)calloc((size_t)W * NFEAT, 8), *src = (int64_t *)calloc((size_t)W, 8), *spk = (int64_t *)calloc((size_t)W, 8),
            *ust = (int64_t *)calloc((size_t)W, 8), dom = h->m.prof[h->prof].dom;
    OrtValue *in[8] = {0};
    int ok = -1;
    if (!pcs || !npc || !wfl || !feats || !src || !spk || !ust) { set_err(h, "out of memory"); goto done; }
    int spksrc = 0;
    for (int i = 0; i < n_all; i++) if (ws[i].spk) { spksrc = 2; break; }
    /* sequential feature state */
    int have_prev = 0; double prev_t1 = 0; int prev_has_t1 = 0; const char *prev_spk = NULL;
    const char *slot_names[16]; int nslots = 0;
    for (int i = 0; i < b; i++) {
        const word_t *w = &ws[i];
        int gap = 0, dur = 0, conf = 0, chg = 0, sp = 0, fd, fg;
        int last_has_t1 = have_prev && prev_has_t1;
        if (w->has_t0) {
            gap = !last_has_t1 ? 9 : (w->t0 - prev_t1 < 0 ? 1 : bk(h->m.gap_e, h->m.n_gap, w->t0 - prev_t1));
            if (w->has_t1) { int L = word_ncp(w->w); dur = bk(h->m.dur_e, h->m.n_dur, (w->t1 - w->t0) / (L > 1 ? L : 1)); if (dur > 7) dur = 7; }
        }
        int fin = w->final ? 1 : 2;
        if (w->has_conf) conf = bk(h->m.conf_e, h->m.n_conf, w->conf);
        if (w->spk) {
            int s = -1;
            for (int k = 0; k < nslots; k++) if (!strcmp(slot_names[k], w->spk)) s = k;
            if (s < 0) { if (nslots < 16) slot_names[nslots] = w->spk; s = nslots < 16 ? nslots++ : 15; }
            sp = s + 1 > 9 ? 9 : s + 1;
            chg = !prev_spk ? 0 : (!strcmp(w->spk, prev_spk) ? 1 : 2);
        }
        if (gap && w->has_t0 && w->has_t1) {
            fd = clampi((int)(rint(w->t1 * 10) - rint(w->t0 * 10) + 1), 1, 20);
            fg = (!have_prev || !prev_has_t1) ? 20 : clampi((int)(rint(w->t0 * 10) - rint(prev_t1 * 10) - 1), 0, 20);
        } else { fd = 3; fg = have_prev ? 0 : 20; }
        if (w->has_t1) { prev_t1 = w->t1; prev_has_t1 = 1; }
        if (w->spk) prev_spk = w->spk;
        have_prev = 1;
        if (i < a) continue;
        int k = i - a, hd, all;
        digit_flags(w->w, &hd, &all);
        for (int p = 0; p < w->npc; p++) pcs[k * P_MAX + p] = w->pcs[p];
        npc[k] = w->npc;
        int64_t fl[5] = {hd, all, w->npc < P_MAX ? w->npc : P_MAX, fd, fg};
        memcpy(wfl + k * 5, fl, sizeof fl);
        int64_t fe[NFEAT] = {gap, dur, fin, conf, chg, h->asr, spksrc, 0};
        memcpy(feats + k * NFEAT, fe, sizeof fe);
        src[k] = (h->asr >= 0 && !strcmp(h->m.asr[h->asr], "arc_transcriber")) ? 1 : 0;
        spk[k] = sp; ust[k] = us[k] ? 1 : 0;
    }
    ust[0] = 1;
    {
        const char *in_names[8] = {"pcs", "npc", "wfl", "feats", "src", "spk", "ustart", "dom"};
        int64_t *bufs[8] = {pcs, npc, wfl, feats, src, spk, ust, &dom};
        int64_t shp[8][2] = {{W, P_MAX}, {W, 0}, {W, 5}, {W, NFEAT}, {W, 0}, {W, 0}, {W, 0}, {1, 0}};
        size_t rank[8] = {2, 1, 2, 2, 1, 1, 1, 1};
        for (int i = 0; i < 8; i++) {
            size_t cnt = (size_t)shp[i][0] * (rank[i] == 2 ? (size_t)shp[i][1] : 1);
            ORT_CHECK(h, h->ort->CreateTensorWithDataAsOrtValue(h->mem, bufs[i], cnt * 8, shp[i], rank[i], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in[i]));
        }
        ORT_CHECK(h, h->ort->Run(h->sess, NULL, in_names, (const OrtValue *const *)in, 8, (const char *const *)h->out_names, (size_t)h->nout, r->vals));
        for (int i = 0; i < h->nout; i++) {
            OrtTensorTypeAndShapeInfo *ti; size_t nd; int64_t dims[4] = {0, 0, 0, 0};
            ORT_CHECK(h, h->ort->GetTensorTypeAndShape(r->vals[i], &ti));
            ORT_CHECK(h, h->ort->GetDimensionsCount(ti, &nd));
            ORT_CHECK(h, h->ort->GetDimensions(ti, dims, nd < 4 ? nd : 4));
            h->ort->ReleaseTensorTypeAndShapeInfo(ti);
            r->dim[i] = nd >= 2 ? (int)dims[1] : 1;
            ORT_CHECK(h, h->ort->GetTensorMutableData(r->vals[i], (void **)&r->o[i]));
        }
    }
    ok = 0;
fail:
done:
    for (int i = 0; i < 8; i++) if (in[i]) h->ort->ReleaseValue(in[i]);
    free(pcs); free(npc); free(wfl); free(feats); free(src); free(spk); free(ust);
    if (ok) run_free(h, r);
    return ok;
}

/* probability of read-out q, row (window-relative) r, column c */
static float P(const arctag_t *h, const run_t *r, int q, int row, int c) {
    int oi = h->qo[q];
    return r->o[oi][(size_t)row * r->dim[oi] + c];
}

/* ------------------------------------------------------------------ JSON helpers */
static cJSON *jnum(double x) { return cJSON_CreateNumber(r6(x)); }

static char *join_words(const word_t *ws, int a, int b) {
    size_t L = 1;
    for (int i = a; i < b; i++) L += strlen(ws[i].w) + 1;
    char *s = (char *)malloc(L); size_t k = 0;
    for (int i = a; i < b; i++) { if (i > a) s[k++] = ' '; size_t n = strlen(ws[i].w); memcpy(s + k, ws[i].w, n); k += n; }
    s[k] = 0;
    return s;
}

/* top-k of a read-out among valid columns whose tag starts with prefix, sorted by -p (stable) */
static cJSON *top(const arctag_t *h, const run_t *r, int q, int row, const char *prefix, int k, int *idx_out, float *p_out, int *n_out) {
    const profile_t *pf = &h->m.prof[h->prof];
    int nc = h->m.nc[q];
    int *idx = (int *)malloc(sizeof(int) * (size_t)(nc + 1)); int n = 0;
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

static double jget(cJSON *o, const char *k) { return cJSON_GetObjectItem(o, k)->valuedouble; }

static void sort_by_p_desc(cJSON *arr) {                                  /* stable */
    int n = cJSON_GetArraySize(arr);
    cJSON **v = (cJSON **)malloc(sizeof(cJSON *) * (size_t)(n + 1));
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
    for (cJSON *x = saf->child; x; x = x->next)
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
    if (s->n == s->cap) { s->cap = s->cap ? 2 * s->cap : 32; s->v = (span_t *)realloc(s->v, sizeof(span_t) * (size_t)s->cap); }
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

/* decode spans over window rows [r0, r1) (global word index = row + off); keep spans with end > a0 */
static cJSON *decode_spans(arctag_t *h, const run_t *r, const unsigned char *us, int r0, int r1, int off, int a0, const word_t *ws) {
    const meta_t *m = &h->m; const profile_t *pf = &m->prof[h->prof];
    int nin = m->nc[Q_IN], W = r1 - r0;
    cJSON *out = cJSON_CreateArray();
    if (W <= 0) return out;
    unsigned char *ON = (unsigned char *)calloc((size_t)W * nin, 1), *BEG = (unsigned char *)calloc((size_t)W * nin, 1);
    unsigned char *headcol = (unsigned char *)calloc((size_t)nin, 1);
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
        int nt = r->dim[it];
        int mem[64], nm = 0;
        for (int t = 0; t < m->fam[f].ntags && nm < 64; t++) {
            int c = find_str(m->cols[Q_IN], nin, m->fam[f].tags[t]);
            if (c >= 0 && headcol[c] && pf->valid[Q_IN][c]) mem[nm++] = t;
        }
        double th = pf->head_thr[f];
        int *lab = (int *)calloc((size_t)W, sizeof(int));
        for (int k = 0; k < W; k++) {
            const float *pb = r->o[ib] + (size_t)(r0 + k) * 3;
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
                    const float *pt = r->o[it] + (size_t)(r0 + x) * nt;
                    for (int t = 0; t < nm; t++) {
                        float v = pt[mem[t]]; if (v < 1e-9f) v = 1e-9f; if (v > 1.0f) v = 1.0f;
                        sc[t] += logf(v); pm[t] += pt[mem[t]];
                    }
                    float pin = 1.0f - r->o[ib][(size_t)(r0 + x) * 3];
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
        char *txt = join_words(ws, s->a, s->b); cJSON_AddStringToObject(o, "text", txt); free(txt);
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
            for (int w = s->a; w < s->b; w++) { int row = w - off; if (row >= 0 && row < r->W) { mean += r->o[vm][row]; cnt++; } }
            int rl = s->b - 1 - off;
            double v = (cnt ? mean / cnt : 0) + (rl >= 0 && rl < r->W ? r->o[ve][rl] : 0);
            cJSON_AddItemToObject(o, "ver", jnum(1.0 / (1.0 + exp(-v))));
        }
        cJSON_AddItemToArray(out, o);
    }
    free(res.v); free(ON); free(BEG); free(headcol);
    return out;
}

/* T0 events of rows [r0, r1); dedup: the same tag within the previous 3 words */
static void decode_events(arctag_t *h, const run_t *r, int r0, int r1, int off, const word_t *ws, cJSON *out) {
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
                if (h->nk == h->kcap) { h->kcap = h->kcap ? 2 * h->kcap : 64; h->keys = (evkey_t *)realloc(h->keys, sizeof(evkey_t) * (size_t)h->kcap); }
                h->keys[h->nk].q = q; h->keys[h->nk].c = c; h->keys[h->nk].i = i; h->nk++;
                cJSON *o = cJSON_CreateObject();
                cJSON_AddStringToObject(o, "type", h->m.cols[q][c]);
                cJSON_AddStringToObject(o, "q", QN[q]);
                cJSON_AddNumberToObject(o, "word", i);
                cJSON_AddItemToObject(o, "score", jnum(p));
                cJSON_AddItemToObject(o, "thr", jnum(pf->thr[q][c]));
                cJSON_AddStringToObject(o, "text", ws[i].w);
                cJSON_AddItemToArray(out, o);
            }
        }
    }
}

/* ------------------------------------------------------------------ words */
static void word_free(word_t *w) { free(w->w); free(w->spk); memset(w, 0, sizeof *w); }

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
    char *buf = (char *)malloc(b - a + 2); buf[0] = ' '; memcpy(buf + 1, s + a, b - a); buf[b - a + 1] = 0;
    int32_t ids[256];
    int k = arctok_encode(h->tok, buf, ids, 256);
    free(buf);
    if (k > 256) {                                                         /* very long word: need the last id too */
        int32_t *big = (int32_t *)malloc(sizeof(int32_t) * (size_t)k);
        char *b2 = (char *)malloc(b - a + 2); b2[0] = ' '; memcpy(b2 + 1, s + a, b - a); b2[b - a + 1] = 0;
        arctok_encode(h->tok, b2, big, k); ids[255] = big[k - 1]; free(big); free(b2);
        for (int i = 0; i < P_MAX - 1; i++) w->pcs[i] = ids[i];
        w->pcs[P_MAX - 1] = ids[255]; w->npc = P_MAX; return;
    }
    if (k <= 0) { w->pcs[0] = 0; w->npc = 1; return; }
    if (k > P_MAX) { for (int i = 0; i < P_MAX - 1; i++) w->pcs[i] = ids[i]; w->pcs[P_MAX - 1] = ids[k - 1]; w->npc = P_MAX; }
    else { for (int i = 0; i < k; i++) w->pcs[i] = ids[i]; w->npc = k; }
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
    memset(w, 0, sizeof *w); w->final = 1; if (eou) *eou = 0;
    static const char *kw[] = {"w", "word", "text"}, *k0[] = {"t0", "start"}, *k1[] = {"t1", "end"}, *kc[] = {"conf", "confidence"};
    if (cJSON_IsString(x)) w->w = xstrdup(x->valuestring);
    else if (cJSON_IsObject(x)) {
        const char *s = first_str(x, kw, 3);
        if (!s) return -1;
        w->w = xstrdup(s);
        w->has_t0 = first_num(x, k0, 2, &w->t0); w->has_t1 = first_num(x, k1, 2, &w->t1); w->has_conf = first_num(x, kc, 2, &w->conf);
        cJSON *sp = cJSON_GetObjectItem(x, "spk"); if (!sp) sp = cJSON_GetObjectItem(x, "speaker");
        if (cJSON_IsString(sp)) w->spk = xstrdup(sp->valuestring);
        else if (cJSON_IsNumber(sp)) { char b[32]; snprintf(b, sizeof b, "%d", sp->valueint); w->spk = xstrdup(b); }
        cJSON *f = cJSON_GetObjectItem(x, "final"); if (cJSON_IsBool(f)) w->final = cJSON_IsTrue(f);
        cJSON *e = cJSON_GetObjectItem(x, "eou"); if (eou && cJSON_IsTrue(e)) *eou = 1;
    } else return -1;
    word_tokenize(h, w);
    return 0;
}

/* ------------------------------------------------------------------ public API */
const char *arctag_version(void) { return ARCTAG_VERSION; }
void arctag_free_str(char *s) { free(s); }
const char *arctag_last_error(const arctag_t *h) { return h ? h->err : ""; }

static char *path_join(const char *d, const char *f) {
    size_t n = strlen(d) + strlen(f) + 2; char *p = (char *)malloc(n);
    snprintf(p, n, "%s%s%s", d, (*d && d[strlen(d) - 1] != '/' && d[strlen(d) - 1] != '\\') ? "/" : "", f);
    return p;
}

arctag_t *arctag_create(const char *model_dir, char *err, size_t errlen) {
    arctag_t *h = (arctag_t *)calloc(1, sizeof(arctag_t));
    if (!h) { if (err && errlen) snprintf(err, errlen, "out of memory"); return NULL; }
    const OrtApiBase *base = OrtGetApiBase();
    h->ort = base ? base->GetApi(ORT_API_VERSION) : NULL;
    char *p;
    if (!h->ort) { set_err(h, "onnxruntime: API version %d not supported by the loaded library", ORT_API_VERSION); goto fail; }
    p = path_join(model_dir, "arctag_meta.json");
    if (load_meta(h, p)) { free(p); goto fail; }
    free(p);
    p = path_join(model_dir, "tokenizer.json");
    h->tok = arctok_load(p, h->err, sizeof h->err); free(p);
    if (!h->tok) goto fail;
    h->prof = h->m.default_prof; h->asr = h->m.default_asr; h->max_ctx = h->m.max_ctx; h->stream_ctx = 256;
    { const char *t = getenv("ARCTAG_THREADS"); h->threads = t ? atoi(t) : 0; }
    for (int q = 0; q < NQ; q++) { h->qo[q] = h->nout; h->out_names[h->nout++] = xstrdup(QN[q]); }
    for (int f = 0; f < h->m.nfam; f++) { h->out_names[h->nout++] = xstrdup(h->m.fam[f].out_t); h->out_names[h->nout++] = xstrdup(h->m.fam[f].out_b); }
    h->out_names[h->nout++] = xstrdup("ver_m"); h->out_names[h->nout++] = xstrdup("ver_e");
    h->model_path = path_join(model_dir, "arctag.onnx");
    ORT_CHECK(h, h->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "arctag", &h->env));
    ORT_CHECK(h, h->ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &h->mem));
    if (open_session(h)) goto fail;
    h->err[0] = 0;
    return h;
fail:
    if (err && errlen) snprintf(err, errlen, "%s", h->err[0] ? h->err : "arctag_create failed");
    arctag_destroy(h);
    return NULL;
}

void arctag_reset(arctag_t *h) {
    if (!h) return;
    for (int i = 0; i < h->n; i++) word_free(&h->words[i]);
    h->n = 0; h->nb = 0; h->np = 0; h->nk = 0; h->lo = 0;
}

void arctag_destroy(arctag_t *h) {
    if (!h) return;
    arctag_reset(h);
    free(h->words); free(h->bounds); free(h->pend); free(h->keys);
    if (h->sess) h->ort->ReleaseSession(h->sess);
    if (h->mem) h->ort->ReleaseMemoryInfo(h->mem);
    if (h->env) h->ort->ReleaseEnv(h->env);
    for (int i = 0; i < h->nout; i++) free(h->out_names[i]);
    if (h->m.prof) free_meta(&h->m);
    arctok_free(h->tok);
    free(h->model_path);
    free(h);
}

int arctag_set_option(arctag_t *h, const char *key, const char *value) {
    if (!h || !key || !value) return -1;
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
    if (!strcmp(key, "threads")) { h->threads = atoi(value); return open_session(h); }
    if (!strcmp(key, "max_ctx")) { int v = atoi(value); if (v < 16 || v > 2048) { set_err(h, "max_ctx must be 16..2048"); return -1; } h->max_ctx = v; return 0; }
    if (!strcmp(key, "stream_ctx")) { int v = atoi(value); if (v < 16 || v > 2048) { set_err(h, "stream_ctx must be 16..2048"); return -1; } h->stream_ctx = v; return 0; }
    set_err(h, "unknown option '%s'", key);
    return -1;
}

static char *print_free(cJSON *o) { char *s = cJSON_PrintUnformatted(o); cJSON_Delete(o); return s; }

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
        unsigned char *u = (unsigned char *)malloc((size_t)W); memcpy(u, us + c0, (size_t)W); u[0] = 1;
        /* features restart at every chunk (= python reference): pass the chunk as its own word list */
        run_t r;
        if (run_model(h, ws + c0, W, 0, W, u, &r)) { free(u); ok = 0; break; }
        append_all(spans, decode_spans(h, &r, u, 0, W, c0, -1, ws));
        decode_events(h, &r, 0, W, c0, ws, events);
        int a = 0;
        for (int i = 1; i <= W; i++) {
            if (i < W && !u[i]) continue;
            cJSON *o = cJSON_CreateObject();
            cJSON_AddNumberToObject(o, "start", a + c0); cJSON_AddNumberToObject(o, "end", i + c0);
            char *t = join_words(ws, a + c0, i + c0); cJSON_AddStringToObject(o, "text", t); free(t);
            int spk_slot = (int)0;
            { /* speaker slot of the last word (same rule as the features) */
                const char *names[16]; int ns = 0, slot = 0;
                for (int k = c0; k < i + c0; k++) if (ws[k].spk) {
                    int s = -1; for (int z = 0; z < ns; z++) if (!strcmp(names[z], ws[k].spk)) s = z;
                    if (s < 0) { if (ns < 16) names[ns] = ws[k].spk; s = ns < 16 ? ns++ : 15; }
                    if (k == i + c0 - 1) slot = s + 1 > 9 ? 9 : s + 1;
                }
                spk_slot = slot;
            }
            cJSON *ro = readouts(h, &r, i - 1, spk_slot);
            for (cJSON *x = ro->child; x;) { cJSON *nx = x->next; cJSON_AddItemToObject(o, x->string, cJSON_DetachItemViaPointer(ro, x)); x = nx; }
            cJSON_Delete(ro);
            cJSON_AddItemToArray(utts, o);
            a = i;
        }
        run_free(h, &r); free(u);
    }
    free(h->keys); h->keys = saved_keys; h->nk = saved_nk; h->kcap = saved_cap;
    if (!ok) { cJSON_Delete(res); return NULL; }
    return print_free(res);
}

static int is_sent_end(const char *w) {
    size_t n = strlen(w);
    while (n && strchr("\"')]}", w[n - 1])) n--;
    return n && strchr(".?!", w[n - 1]) != NULL;
}

char *arctag_tag_text(arctag_t *h, const char *text) {
    if (!h || !text) return NULL;
    int cap = 64, n = 0;
    word_t *ws = (word_t *)calloc((size_t)cap, sizeof(word_t));
    unsigned char *us = (unsigned char *)calloc((size_t)cap, 1);
    const char *s = text; int nxt = 1;
    while (*s) {
        while (*s && strchr(" \t\n\r\f\v", *s)) s++;
        if (!*s) break;
        const char *e = s; while (*e && !strchr(" \t\n\r\f\v", *e)) e++;
        if (n == cap) { cap *= 2; ws = (word_t *)realloc(ws, sizeof(word_t) * (size_t)cap); us = (unsigned char *)realloc(us, (size_t)cap); }
        memset(&ws[n], 0, sizeof(word_t));
        ws[n].w = (char *)malloc((size_t)(e - s) + 1); memcpy(ws[n].w, s, (size_t)(e - s)); ws[n].w[e - s] = 0; ws[n].final = 1;
        word_tokenize(h, &ws[n]);
        us[n] = (unsigned char)nxt; nxt = is_sent_end(ws[n].w);
        n++; s = e;
    }
    char *out = tag_words_impl(h, ws, n, us);
    for (int i = 0; i < n; i++) word_free(&ws[i]);
    free(ws); free(us);
    return out;
}

char *arctag_tag_words(arctag_t *h, const char *words_json) {
    if (!h || !words_json) return NULL;
    cJSON *j = cJSON_Parse(words_json);
    if (!cJSON_IsArray(j)) { set_err(h, "tag_words: expected a JSON array"); cJSON_Delete(j); return NULL; }
    int n = cJSON_GetArraySize(j), k = 0, nxt = 1;
    word_t *ws = (word_t *)calloc((size_t)(n + 1), sizeof(word_t));
    unsigned char *us = (unsigned char *)calloc((size_t)(n + 1), 1);
    for (cJSON *x = j->child; x; x = x->next) {
        int eou;
        if (parse_word(h, x, &ws[k], &eou)) { set_err(h, "tag_words: bad word at %d", k); continue; }
        us[k] = (unsigned char)nxt; nxt = eou; k++;
    }
    cJSON_Delete(j);
    char *out = tag_words_impl(h, ws, k, us);
    for (int i = 0; i < k; i++) word_free(&ws[i]);
    free(ws); free(us);
    return out;
}

static void add_bound(arctag_t *h) {
    if (!h->n || (h->nb && h->bounds[h->nb - 1] >= h->n)) return;
    if (h->nb == h->bcap) { h->bcap = h->bcap ? 2 * h->bcap : 64; h->bounds = (int *)realloc(h->bounds, sizeof(int) * (size_t)h->bcap); }
    if (h->np == h->pcap) { h->pcap = h->pcap ? 2 * h->pcap : 64; h->pend = (int *)realloc(h->pend, sizeof(int) * (size_t)h->pcap); }
    h->pend[h->np++] = h->nb;
    h->bounds[h->nb++] = h->n;
}

int arctag_push_words(arctag_t *h, const char *words_json) {
    if (!h || !words_json) return -1;
    cJSON *j = cJSON_Parse(words_json);
    if (!j) { set_err(h, "push_words: invalid JSON"); return -1; }
    cJSON *arr = NULL; int from = h->n, endpoint = 0;
    if (cJSON_IsArray(j)) arr = j;
    else if (cJSON_IsObject(j)) {
        arr = cJSON_GetObjectItem(j, "words");
        cJSON *f = cJSON_GetObjectItem(j, "from"); if (cJSON_IsNumber(f)) from = f->valueint;
        endpoint = cJSON_IsTrue(cJSON_GetObjectItem(j, "endpoint"));
    } else { set_err(h, "push_words: expected an array or an object"); cJSON_Delete(j); return -1; }
    if (from < 0) from = 0;
    if (from > h->n) from = h->n;
    if (arr && from < h->n) {                                       /* revision: drop words[from:] and later boundaries */
        for (int i = from; i < h->n; i++) word_free(&h->words[i]);
        h->n = from;
        while (h->nb && h->bounds[h->nb - 1] > from) h->nb--;
        int k = 0; for (int i = 0; i < h->np; i++) if (h->pend[i] < h->nb) h->pend[k++] = h->pend[i];
        h->np = k;
    }
    if (from < h->lo) h->lo = from;
    for (cJSON *x = arr ? arr->child : NULL; x; x = x->next) {
        if (h->n == h->cap) { h->cap = h->cap ? 2 * h->cap : 256; h->words = (word_t *)realloc(h->words, sizeof(word_t) * (size_t)h->cap); }
        int eou;
        if (parse_word(h, x, &h->words[h->n], &eou)) { set_err(h, "push_words: bad word"); cJSON_Delete(j); return -1; }
        h->n++;
        if (eou) add_bound(h);
    }
    if (endpoint) add_bound(h);
    cJSON_Delete(j);
    return 0;
}

char *arctag_poll(arctag_t *h) {
    if (!h) return NULL;
    int n = h->n;
    cJSON *res = cJSON_CreateObject();
    cJSON_AddNumberToObject(res, "n_words", n);
    int a0 = n - h->m.span_win > 0 ? n - h->m.span_win : 0;
    cJSON_AddNumberToObject(res, "a0", a0);
    if (!n) {
        cJSON_AddArrayToObject(res, "spans"); cJSON_AddArrayToObject(res, "events"); cJSON_AddArrayToObject(res, "utterances");
        cJSON_AddNullToObject(res, "current");
        return print_free(res);
    }
    int s = n - h->stream_ctx > 0 ? n - h->stream_ctx : 0, W = n - s;
    unsigned char *us = (unsigned char *)calloc((size_t)W, 1);
    us[0] = 1;
    for (int b = 0; b < h->nb; b++) if (h->bounds[b] >= s && h->bounds[b] < n) us[h->bounds[b] - s] = 1;
    run_t r;
    if (run_model(h, h->words, n, s, n, us, &r)) { free(us); cJSON_Delete(res); return NULL; }
    int d0 = a0 - 16 > s ? a0 - 16 : s;
    cJSON_AddItemToObject(res, "spans", decode_spans(h, &r, us, d0 - s, W, s, a0, h->words));
    cJSON *ev = cJSON_AddArrayToObject(res, "events");
    int lo = h->lo > s ? h->lo : s;
    decode_events(h, &r, lo - s, W, s, h->words, ev);
    h->lo = n;
    cJSON *utts = cJSON_AddArrayToObject(res, "utterances");
    for (int i = 0; i < h->np; i++) {
        int bi = h->pend[i], end = h->bounds[bi], start = bi ? h->bounds[bi - 1] : 0;
        if (end - 1 < s) continue;                                   /* fell out of the window: skipped */
        cJSON *o = cJSON_CreateObject();
        cJSON_AddNumberToObject(o, "start", start); cJSON_AddNumberToObject(o, "end", end);
        char *t = join_words(h->words, start, end); cJSON_AddStringToObject(o, "text", t); free(t);
        cJSON_AddBoolToObject(o, "final", 1);
        int slot = 0;
        { const char *names[16]; int ns = 0;
          for (int k = 0; k < end; k++) if (h->words[k].spk) {
              int z = -1; for (int y = 0; y < ns; y++) if (!strcmp(names[y], h->words[k].spk)) z = y;
              if (z < 0) { if (ns < 16) names[ns] = h->words[k].spk; z = ns < 16 ? ns++ : 15; }
              if (k == end - 1) slot = z + 1 > 9 ? 9 : z + 1;
          } }
        cJSON *ro = readouts(h, &r, end - 1 - s, slot);
        for (cJSON *x = ro->child; x;) { cJSON *nx = x->next; cJSON_AddItemToObject(o, x->string, cJSON_DetachItemViaPointer(ro, x)); x = nx; }
        cJSON_Delete(ro);
        cJSON_AddItemToArray(utts, o);
    }
    h->np = 0;
    int ustart = h->nb ? h->bounds[h->nb - 1] : 0;
    if (ustart < n) {
        cJSON *o = cJSON_CreateObject();
        cJSON_AddNumberToObject(o, "start", ustart); cJSON_AddNumberToObject(o, "end", n);
        cJSON_AddBoolToObject(o, "final", 0);
        int slot = 0;
        { const char *names[16]; int ns = 0;
          for (int k = 0; k < n; k++) if (h->words[k].spk) {
              int z = -1; for (int y = 0; y < ns; y++) if (!strcmp(names[y], h->words[k].spk)) z = y;
              if (z < 0) { if (ns < 16) names[ns] = h->words[k].spk; z = ns < 16 ? ns++ : 15; }
              if (k == n - 1) slot = z + 1 > 9 ? 9 : z + 1;
          } }
        cJSON *ro = readouts(h, &r, W - 1, slot);
        for (cJSON *x = ro->child; x;) { cJSON *nx = x->next; cJSON_AddItemToObject(o, x->string, cJSON_DetachItemViaPointer(ro, x)); x = nx; }
        cJSON_Delete(ro);
        cJSON_AddItemToObject(res, "current", o);
    } else cJSON_AddNullToObject(res, "current");
    run_free(h, &r); free(us);
    return print_free(res);
}
