/* arctag: Arc Tagger C library. Two backends run the model: native (plain C, src/native.c) and ONNX Runtime (built with
 * ARCTAG_WITH_ORT). Features, streaming, decoding and JSON are shared. */
#include "arctag.h"
#include "tokenizer.h"
#include "native.h"
#include "cJSON.h"
#ifdef ARCTAG_WITH_ORT
#include <onnxruntime_c_api.h>
#endif
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define ARCTAG_VERSION "0.3.0"
#define NQ 10
#define P_MAX 8
#define NFEAT 8
#define MAX_FAM 8
#define MAX_OUT 32
#define MAX_SEG 16
#define N_RCAT 11

static const char *QN[NQ] = {"frame", "span_beg", "span_in", "span_end", "ev_at", "ev_imm", "ev_rec", "utt", "speaker", "item"};
enum { Q_FRAME, Q_BEG, Q_IN, Q_END, Q_EVAT, Q_EVIMM, Q_EVREC, Q_UTT, Q_SPK, Q_ITEM };

/* ------------------------------------------------------------------ small utils */
static char *xstrdup(const char *s) { size_t n = strlen(s) + 1; char *c = (char *)malloc(n); if (c) memcpy(c, s, n); return c; }
static double r6(double x) { return round(x * 1e6) / 1e6; }
static int starts_with(const char *s, const char *p) { return strncmp(s, p, strlen(p)) == 0; }

/* ------------------------------------------------------------------ model metadata */
typedef struct { char *name, *ui, *fine; int dom; double *thr[NQ]; unsigned char *valid[NQ]; double head_thr[MAX_FAM]; } profile_t;
typedef struct { char *name, *out_t, *out_b; int ntags; char **tags; int use; } fam_t;
typedef struct { char *ui; int ncue; int *ntok; char *(*tok)[3]; int *fine; int nfine; char **fines; } cues_t;   /* fine-domain lexicon cues */

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
    unsigned char *ev_skip[NQ];           /* event columns not displayed at T0 (hidden, or another display tier) */
    int ncues; cues_t *cues; int fine_min, fine_every; double fine_ratio;
} meta_t;

/* ------------------------------------------------------------------ words / stream */
typedef struct {
    char *w; double t0, t1, conf; int has_t0, has_t1, has_conf, final; char *spk;
    int npc; int64_t pcs[P_MAX];
} word_t;

typedef struct { int q, c, i; } evkey_t;
typedef struct { char *pat; int q; double v; } thr_ov_t;      /* threshold override; q = -1: every read-out, -2: head family */

/* output sections (option "emit") */
enum { E_ENTITY, E_NUMBER, E_PII, E_PROF, E_TOX, E_EVENTS, E_INTENT, E_ACT, E_SAFETY, E_EMOTION, E_AROUSAL, E_VALENCE, E_INTENSITY,
       E_ROLE, E_ITEM, E_STATE, N_EMIT };
static const char *EMIT_N[N_EMIT] = {"entity", "number", "pii", "profanity", "toxicity", "events", "intent", "act", "safety", "emotion",
                                     "arousal", "valence", "intensity", "role", "item", "state"};
/* redaction categories */
static const char *RCAT[N_RCAT] = {"phone", "email", "address", "card", "gov_id", "dob", "name", "handle", "contact", "profanity", "toxicity"};
static const char *RLAB[N_RCAT] = {"PHONE", "EMAIL", "ADDRESS", "CARD", "GOV_ID", "DOB", "NAME", "HANDLE", "CONTACT", "PROFANITY", "TOXIC"};

struct arctag {
#ifdef ARCTAG_WITH_ORT
    const OrtApi *ort; OrtEnv *env; OrtSession *sess; OrtMemoryInfo *mem;
#endif
    nat_model *nm; nat *one, *st;          /* native backend: model, one-pass state, streaming state */
    int noff[MAX_OUT], ndim[MAX_OUT], ntotal;
    float *rows; int rcap, rbase, sdone, sdirty, svalid, st_cap, st_dom, st_spk, st_asr, stream_state;
    char *model_dir, *model_path, *model_file; int threads, precision, backend;
    meta_t m; arctok_t *tok;
    int prof, base_prof, asr, max_ctx, stream_ctx;
    double *thr[NQ]; double head[MAX_FAM];   /* thresholds of the active profile with the overrides applied */
    thr_ov_t *ov; int nov;
    int fine_auto, fine_chk;
    unsigned char emit[N_EMIT]; int all_emit; char **tagpre; int ntagpre;
    int redact; unsigned char rpol[N_RCAT];  /* redact: 0 off, 1 on; rpol: 1 = mask */
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
    for (int i = 0; i < m->nasr; i++)                 /* human-written text: the model files call it "gold"; the API says "human" */
        if (!strcmp(m->asr[i], "gold")) { free(m->asr[i]); m->asr[i] = xstrdup("human"); }
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
        cJSON *ui = cJSON_GetObjectItem(p, "ui"), *fi = cJSON_GetObjectItem(p, "fine");
        P->ui = xstrdup(cJSON_IsString(ui) ? ui->valuestring : ""); P->fine = xstrdup(cJSON_IsString(fi) ? fi->valuestring : "");
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
    /* events not displayed at T0: hidden ones and those whose display tier is another tier */
    cJSON *ev = cJSON_GetObjectItem(j, "events"), *evt = cJSON_GetObjectItem(ev, "tier"), *evh = cJSON_GetObjectItem(ev, "hide");
    for (int q = 0; q < NQ; q++) {
        m->ev_skip[q] = (unsigned char *)calloc((size_t)m->nc[q] + 1, 1);
        for (int c = 0; c < m->nc[q]; c++) {
            cJSON *t = cJSON_GetObjectItem(evt, m->cols[q][c]);
            if (cJSON_IsString(t) && strcmp(t->valuestring, "T0")) m->ev_skip[q][c] = 1;
            for (cJSON *hx = evh ? evh->child : NULL; hx; hx = hx->next) if (cJSON_IsString(hx) && !strcmp(hx->valuestring, m->cols[q][c])) m->ev_skip[q][c] = 1;
        }
    }
    /* fine-domain lexicon cues {ui: {fine: ["cue", "two-word-cue", ...]}} */
    cJSON *fc = cJSON_GetObjectItem(j, "fine_cues");
    m->ncues = cJSON_GetArraySize(fc);
    m->cues = (cues_t *)calloc((size_t)m->ncues + 1, sizeof(cues_t));
    int ci = 0;
    for (cJSON *u = fc ? fc->child : NULL; u; u = u->next, ci++) {
        cues_t *C = &m->cues[ci]; C->ui = xstrdup(u->string);
        int tot = 0; C->nfine = cJSON_GetArraySize(u);
        for (cJSON *f = u->child; f; f = f->next) tot += cJSON_GetArraySize(f);
        C->ntok = (int *)calloc((size_t)tot + 1, sizeof(int)); C->tok = (char *(*)[3])calloc((size_t)tot + 1, sizeof *C->tok);
        C->fine = (int *)calloc((size_t)tot + 1, sizeof(int)); C->fines = (char **)calloc((size_t)C->nfine + 1, sizeof(char *));
        int fi = 0;
        for (cJSON *f = u->child; f; f = f->next, fi++) {
            C->fines[fi] = xstrdup(f->string);
            for (cJSON *c = f->child; c; c = c->next) {
                if (!cJSON_IsString(c)) continue;
                char *t[3] = {0}; int nt = 0; char buf[128]; snprintf(buf, sizeof buf, "%s", c->valuestring);
                for (char *x = buf; *x; x++) if (*x == '-') *x = ' ';
                for (char *x = strtok(buf, " "); x && nt < 3; x = strtok(NULL, " ")) t[nt++] = xstrdup(x);
                int dup = -1;                                        /* python dict: a later fine overwrites the same cue */
                for (int k = 0; k < C->ncue && dup < 0; k++) {
                    if (C->ntok[k] != nt) continue;
                    int same = 1; for (int z = 0; z < nt; z++) if (strcmp(C->tok[k][z], t[z])) same = 0;
                    if (same) dup = k;
                }
                if (dup >= 0) { C->fine[dup] = fi; for (int z = 0; z < nt; z++) free(t[z]); continue; }
                C->ntok[C->ncue] = nt; for (int z = 0; z < 3; z++) C->tok[C->ncue][z] = t[z];
                C->fine[C->ncue++] = fi;
            }
        }
    }
    x = cJSON_GetObjectItem(j, "fine_min"); m->fine_min = cJSON_IsNumber(x) ? x->valueint : 4;
    x = cJSON_GetObjectItem(j, "fine_ratio"); m->fine_ratio = cJSON_IsNumber(x) ? x->valuedouble : 2.0;
    x = cJSON_GetObjectItem(j, "fine_every"); m->fine_every = cJSON_IsNumber(x) && x->valueint > 0 ? x->valueint : 25;
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
    for (int q = 0; q < NQ; q++) free(m->ev_skip[q]);
    for (int i = 0; i < m->ncues; i++) {
        cues_t *C = &m->cues[i];
        for (int k = 0; k < C->ncue; k++) for (int z = 0; z < 3; z++) free(C->tok[k][z]);
        for (int f = 0; f < C->nfine; f++) free(C->fines[f]);
        free(C->ui); free(C->ntok); free(C->tok); free(C->fine); free(C->fines);
    }
    free(m->cues);
    for (int p = 0; p < m->nprof; p++) {
        free(m->prof[p].name); free(m->prof[p].ui); free(m->prof[p].fine);
        for (int q = 0; q < NQ; q++) { free(m->prof[p].thr[q]); free(m->prof[p].valid[q]); }
    }
    free(m->prof);
    for (int i = 0; i < m->nalias; i++) free(m->alias_k[i]);
    free(m->alias_k); free(m->alias_v);
}

/* ------------------------------------------------------------------ thresholds (profile + overrides) */
static int pat_match(const char *pat, const char *tag) {
    size_t n = strlen(pat);
    if (n && pat[n - 1] == '*') return strncmp(tag, pat, n - 1) == 0;
    return strcmp(tag, pat) == 0;
}

/* make profile p the active one: copy its thresholds and apply the overrides in order */
static void use_profile(arctag_t *h, int p) {
    const meta_t *m = &h->m; const profile_t *pf = &m->prof[p];
    h->prof = p;
    for (int q = 0; q < NQ; q++) {
        if (!h->thr[q]) h->thr[q] = (double *)calloc((size_t)m->nc[q] + 1, sizeof(double));
        memcpy(h->thr[q], pf->thr[q], sizeof(double) * (size_t)m->nc[q]);
    }
    for (int f = 0; f < m->nfam; f++) h->head[f] = pf->head_thr[f];
    for (int k = 0; k < h->nov; k++) {
        const thr_ov_t *o = &h->ov[k];
        if (o->q == -2) { for (int f = 0; f < m->nfam; f++) if (!strcmp(m->fam[f].name, o->pat)) h->head[f] = o->v; continue; }
        for (int q = 0; q < NQ; q++) {
            if (o->q >= 0 && o->q != q) continue;
            for (int c = 0; c < m->nc[q]; c++) if (pat_match(o->pat, m->cols[q][c])) h->thr[q][c] = o->v;
        }
        if (o->q == -1) {
            size_t n = strlen(o->pat);
            for (int f = 0; f < m->nfam; f++) {
                char fp[128]; snprintf(fp, sizeof fp, "%s.", m->fam[f].name);
                if (!strcmp(o->pat, m->fam[f].name) || (n && o->pat[n - 1] == '*' && !strncmp(fp, o->pat, n - 1))) h->head[f] = o->v;
            }
        }
    }
}

static int find_profile(const meta_t *m, const char *ui, const char *fine) {
    for (int p = 0; p < m->nprof; p++) if (!strcmp(m->prof[p].ui, ui) && !strcmp(m->prof[p].fine, fine)) return p;
    return -1;
}

/* the fine domain the words' lexicon cues point to inside UI domain ui: profile index or -1 */
static int fine_hint(const arctag_t *h, const char *ui, const char *const *words, int n) {
    const meta_t *m = &h->m; const cues_t *C = NULL;
    for (int i = 0; i < m->ncues; i++) if (!strcmp(m->cues[i].ui, ui)) C = &m->cues[i];
    if (!C || n <= 0) return -1;
    int a = n > 3000 ? n - 3000 : 0, W = n - a;
    char **tk = (char **)calloc((size_t)W + 1, sizeof(char *));
    for (int i = 0; i < W; i++) {                       /* lower-case, keep [a-z0-9] */
        const char *w = words[a + i]; size_t L = strlen(w), k = 0;
        tk[i] = (char *)malloc(L + 1);
        for (size_t z = 0; z < L; z++) {
            char c = w[z];
            if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) tk[i][k++] = c;
        }
        tk[i][k] = 0;
    }
    int cnt[64] = {0};
    for (int i = 0; i < W; i++)
        for (int L = 1; L <= 3; L++) {
            if (i + L > W) continue;
            for (int k = 0; k < C->ncue; k++) {
                if (C->ntok[k] != L) continue;
                int same = 1; for (int z = 0; z < L && same; z++) if (strcmp(C->tok[k][z], tk[i + z])) same = 0;
                if (same && C->fine[k] < 64) { cnt[C->fine[k]]++; break; }
            }
        }
    for (int i = 0; i < W; i++) free(tk[i]);
    free(tk);
    int b1 = -1, b2 = -1;
    for (int f = 0; f < C->nfine && f < 64; f++) {
        if (!cnt[f]) continue;
        if (b1 < 0 || cnt[f] > cnt[b1]) { b2 = b1; b1 = f; } else if (b2 < 0 || cnt[f] > cnt[b2]) b2 = f;
    }
    if (b1 < 0) return -1;
    if (cnt[b1] >= m->fine_min && (b2 < 0 || cnt[b1] >= m->fine_ratio * cnt[b2])) return find_profile(m, ui, C->fines[b1]);
    return -1;
}

/* ------------------------------------------------------------------ backends
 * Both backends produce the same thing: the graph's outputs for words [a, b) as float rows (run_t). The native backend also
 * keeps a streaming state between polls (nat_window), so a poll only computes the words that are new since the last one. */
typedef struct {
    int W; float *o[MAX_OUT]; int dim[MAX_OUT], st[MAX_OUT];   /* output i, row r, column c: o[i][r * st[i] + c] */
    float *own;                                                /* native: rows owned by this result (NULL for the stream cache) */
#ifdef ARCTAG_WITH_ORT
    OrtValue *vals[MAX_OUT];
#endif
} run_t;

static void run_free(arctag_t *h, run_t *r) {
#ifdef ARCTAG_WITH_ORT
    for (int i = 0; i < h->nout; i++) if (r->vals[i]) h->ort->ReleaseValue(r->vals[i]);
#else
    (void)h;
#endif
    free(r->own);
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

/* model inputs, one word per row */
typedef struct { int64_t pcs[P_MAX], npc, wfl[5], feats[NFEAT], src, spk, ust; } inrow_t;

static int any_speaker(const word_t *ws, int n) { for (int i = 0; i < n; i++) if (ws[i].spk) return 1; return 0; }

/* inputs for words[a..b) (features computed sequentially from word 0); us[k] = 1 if word a + k starts an utterance.
 * The first word of a pass always starts one: force_first. */
static void build_inputs(const arctag_t *h, const word_t *ws, int n_all, int a, int b, const unsigned char *us, int force_first, inrow_t *in) {
    int spksrc = any_speaker(ws, n_all) ? 2 : 0;
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
        inrow_t *x = &in[k];
        memset(x, 0, sizeof *x);
        digit_flags(w->w, &hd, &all);
        for (int p = 0; p < w->npc; p++) x->pcs[p] = w->pcs[p];
        x->npc = w->npc;
        int64_t fl[5] = {hd, all, w->npc < P_MAX ? w->npc : P_MAX, fd, fg};
        memcpy(x->wfl, fl, sizeof fl);
        int64_t fe[NFEAT] = {gap, dur, fin, conf, chg, h->asr, spksrc, 0};
        memcpy(x->feats, fe, sizeof fe);
        x->src = (h->asr >= 0 && !strcmp(h->m.asr[h->asr], "arc_transcriber")) ? 1 : 0;
        x->spk = sp; x->ust = us[k] ? 1 : 0;
    }
    if (force_first && b > a) in[0].ust = 1;
}

static void nat_input(const inrow_t *x, nat_word *o) {
    memset(o, 0, sizeof *o);
    for (int p = 0; p < P_MAX && p < 8; p++) o->pcs[p] = (int)x->pcs[p];
    o->npc = (int)x->npc;
    for (int k = 0; k < 5; k++) o->wfl[k] = (int)x->wfl[k];
    for (int k = 0; k < NFEAT && k < 8; k++) o->feats[k] = (int)x->feats[k];
    o->src = (int)x->src; o->spk = (int)x->spk; o->ust = (int)x->ust;
}

/* point a result at native rows (stride = all outputs of a word) */
static void nat_view(const arctag_t *h, float *rows, int W, run_t *r) {
    r->W = W;
    for (int i = 0; i < h->nout; i++) { r->o[i] = rows + h->noff[i]; r->dim[i] = h->ndim[i]; r->st[i] = h->ntotal; }
}

#ifdef ARCTAG_WITH_ORT
#define ORT_CHECK(h, expr) do { OrtStatus *st_ = (expr); if (st_) { set_err(h, "onnxruntime: %s", (h)->ort->GetErrorMessage(st_)); (h)->ort->ReleaseStatus(st_); goto fail; } } while (0)

static int open_ort(arctag_t *h) {
    OrtSessionOptions *so = NULL; OrtSession *ns = NULL;
    if (!h->ort) {
        const OrtApiBase *base = OrtGetApiBase();
        h->ort = base ? base->GetApi(ORT_API_VERSION) : NULL;
        if (!h->ort) { set_err(h, "onnxruntime: API version %d not supported by the loaded library", ORT_API_VERSION); return -1; }
    }
    if (!h->env) ORT_CHECK(h, h->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "arctag", &h->env));
    if (!h->mem) ORT_CHECK(h, h->ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &h->mem));
    ORT_CHECK(h, h->ort->CreateSessionOptions(&so));
    ORT_CHECK(h, h->ort->SetIntraOpNumThreads(so, h->threads));
    ORT_CHECK(h, h->ort->SetInterOpNumThreads(so, 1));
    ORT_CHECK(h, h->ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
    /* A model stored with int8 weight tables (tools/quantize_model.py) has DequantizeLinear nodes on constant inputs. By
     * default ONNX Runtime keeps them for its quantised-graph passes and runs them on every call; with those passes off
     * they fold into float tables once, at load. */
    ORT_CHECK(h, h->ort->AddSessionConfigEntry(so, "session.disable_quant_qdq", "1"));
#ifdef _WIN32
    {
        int n = MultiByteToWideChar(CP_UTF8, 0, h->model_path, -1, NULL, 0);
        wchar_t *wp = (wchar_t *)malloc(sizeof(wchar_t) * (size_t)n);
        MultiByteToWideChar(CP_UTF8, 0, h->model_path, -1, wp, n);
        OrtStatus *st = h->ort->CreateSession(h->env, wp, so, &ns);
        free(wp);
        ORT_CHECK(h, st);
    }
#else
    ORT_CHECK(h, h->ort->CreateSession(h->env, h->model_path, so, &ns));
#endif
    h->ort->ReleaseSessionOptions(so);
    if (h->sess) h->ort->ReleaseSession(h->sess);       /* swap only once the new session loaded */
    h->sess = ns;
    return 0;
fail:
    if (so) h->ort->ReleaseSessionOptions(so);
    return -1;
}

static void close_ort(arctag_t *h) {
    if (!h->ort) return;
    if (h->sess) h->ort->ReleaseSession(h->sess);
    if (h->mem) h->ort->ReleaseMemoryInfo(h->mem);
    if (h->env) h->ort->ReleaseEnv(h->env);
    h->sess = NULL; h->mem = NULL; h->env = NULL;
}

static int run_ort(arctag_t *h, const inrow_t *rows, int W, run_t *r) {
    int64_t *pcs = (int64_t *)calloc((size_t)W * P_MAX, 8), *npc = (int64_t *)calloc((size_t)W, 8), *wfl = (int64_t *)calloc((size_t)W * 5, 8),
            *feats = (int64_t *)calloc((size_t)W * NFEAT, 8), *src = (int64_t *)calloc((size_t)W, 8), *spk = (int64_t *)calloc((size_t)W, 8),
            *ust = (int64_t *)calloc((size_t)W, 8), dom = h->m.prof[h->prof].dom;
    OrtValue *in[8] = {0};
    int ok = -1;
    if (!pcs || !npc || !wfl || !feats || !src || !spk || !ust) { set_err(h, "out of memory"); goto done; }
    for (int k = 0; k < W; k++) {
        memcpy(pcs + k * P_MAX, rows[k].pcs, sizeof rows[k].pcs); npc[k] = rows[k].npc;
        memcpy(wfl + k * 5, rows[k].wfl, sizeof rows[k].wfl); memcpy(feats + k * NFEAT, rows[k].feats, sizeof rows[k].feats);
        src[k] = rows[k].src; spk[k] = rows[k].spk; ust[k] = rows[k].ust;
    }
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
            r->dim[i] = nd >= 2 ? (int)dims[1] : 1; r->st[i] = r->dim[i];
            ORT_CHECK(h, h->ort->GetTensorMutableData(r->vals[i], (void **)&r->o[i]));
        }
    }
    ok = 0;
fail:
done:
    for (int i = 0; i < 8; i++) if (in[i]) h->ort->ReleaseValue(in[i]);
    free(pcs); free(npc); free(wfl); free(feats); free(src); free(spk); free(ust);
    return ok;
}
#endif

/* native, one pass over W words from a fresh state (the same as one ONNX Runtime run) */
static int run_native(arctag_t *h, const inrow_t *rows, int W, run_t *r) {
    int dom = h->m.prof[h->prof].dom;
    if (!h->one && !(h->one = nat_new(h->nm, dom, h->m.max_ctx > 0 ? h->m.max_ctx : 2048, h->m.max_ctx > 0 ? h->m.max_ctx : 2048))) { set_err(h, "out of memory"); return -1; }
    if (W > h->m.max_ctx && h->m.max_ctx > 0) { set_err(h, "pass of %d words is longer than the model's context", W); return -1; }
    nat_word *in = (nat_word *)malloc(sizeof(nat_word) * (size_t)(W > 0 ? W : 1));
    r->own = (float *)malloc(sizeof(float) * (size_t)h->ntotal * (size_t)(W > 0 ? W : 1));
    if (!in || !r->own) { free(in); free(r->own); r->own = NULL; set_err(h, "out of memory"); return -1; }
    for (int k = 0; k < W; k++) nat_input(&rows[k], &in[k]);
    if (nat_set_domain(h->one, dom)) { free(in); free(r->own); r->own = NULL; set_err(h, "domain %d out of range", dom); return -1; }
    nat_set_snapshots(h->one, 0);
    nat_steps(h->one, in, W, r->own);
    free(in);
    nat_view(h, r->own, W, r);
    if (W > 512) { nat_free(h->one); h->one = NULL; }   /* do not keep a long pass's memory */
    return 0;
}

/* model outputs for words[a..b) in one pass from a fresh state, us = utterance starts (us[0] is forced on) */
static int run_model(arctag_t *h, const word_t *ws, int n_all, int a, int b, const unsigned char *us, run_t *r) {
    int W = b - a;
    memset(r, 0, sizeof *r); r->W = W;
    inrow_t *rows = (inrow_t *)malloc(sizeof(inrow_t) * (size_t)(W > 0 ? W : 1));
    if (!rows) { set_err(h, "out of memory"); return -1; }
    build_inputs(h, ws, n_all, a, b, us, 1, rows);
    int rc;
#ifdef ARCTAG_WITH_ORT
    if (h->backend == ARCTAG_BACKEND_ONNXRUNTIME) rc = run_ort(h, rows, W, r);
    else
#endif
    rc = run_native(h, rows, W, r);
    free(rows);
    if (rc) run_free(h, r);
    return rc;
}

/* ---- native streaming: the state continues from poll to poll; rows of words [rbase, sdone) are cached */
static void stream_invalidate(arctag_t *h) { h->sdone = h->rbase = 0; h->sdirty = 0; h->svalid = 0; }

static int rows_reserve(arctag_t *h, int nrows) {
    if (nrows <= h->rcap) return 0;
    int c = h->rcap ? h->rcap : 256; while (c < nrows) c *= 2;
    float *p = (float *)realloc(h->rows, sizeof(float) * (size_t)h->ntotal * (size_t)c);
    if (!p) { set_err(h, "out of memory"); return -1; }
    h->rows = p; h->rcap = c;
    return 0;
}

/* rows for words[s..n) of the stream, from the state where possible */
static int nat_window(arctag_t *h, int s, int n, run_t *r) {
    int dom = h->m.prof[h->prof].dom, spksrc = any_speaker(h->words, n);
    memset(r, 0, sizeof *r);
    if (!h->st || h->st_cap != h->stream_ctx) {
        nat_free(h->st); h->st = nat_new(h->nm, dom, h->stream_ctx + NAT_REWIND_WORDS, h->stream_ctx);
        if (!h->st) { set_err(h, "out of memory"); return -1; }
        h->st_cap = h->stream_ctx; h->st_dom = dom; h->svalid = 0;
    }
    int restart = !h->svalid || h->st_dom != dom || h->st_spk != spksrc || h->st_asr != h->asr || !h->stream_state
                  || nat_first(h->st) > s || h->sdone > n;
    if (!restart && h->sdirty < h->sdone) {                 /* a revision: back to a snapshot at or before it */
        int p = nat_rewind(h->st, h->sdirty);
        if (p < 0 || p < h->rbase) restart = 1; else h->sdone = p;
    }
    if (restart) {
        if (h->st_dom != dom && nat_set_domain(h->st, dom)) { set_err(h, "domain %d out of range", dom); return -1; }
        nat_restart(h->st, s);
        h->st_dom = dom; h->st_spk = spksrc; h->st_asr = h->asr;
        h->rbase = h->sdone = s; h->svalid = 1;
    }
    h->sdirty = INT32_MAX;
    int a = h->sdone, todo = n - a;
    if (todo > 0) {
        if (rows_reserve(h, n - h->rbase)) { h->svalid = 0; return -1; }
        inrow_t *in = (inrow_t *)malloc(sizeof(inrow_t) * (size_t)todo);
        nat_word *nw = (nat_word *)malloc(sizeof(nat_word) * (size_t)todo);
        unsigned char *us = (unsigned char *)calloc((size_t)todo, 1);
        if (!in || !nw || !us) { free(in); free(nw); free(us); h->svalid = 0; set_err(h, "out of memory"); return -1; }
        for (int b = 0; b < h->nb; b++) if (h->bounds[b] >= a && h->bounds[b] < n) us[h->bounds[b] - a] = 1;
        build_inputs(h, h->words, n, a, n, us, a == nat_first(h->st), in);
        for (int k = 0; k < todo; k++) nat_input(&in[k], &nw[k]);
        nat_set_snapshots(h->st, 1);
        nat_steps(h->st, nw, todo, h->rows + (size_t)(a - h->rbase) * h->ntotal);
        free(in); free(nw); free(us);
        h->sdone = n;
    }
    if (s - h->rbase > 256) {                                /* drop rows older than the window */
        memmove(h->rows, h->rows + (size_t)(s - h->rbase) * h->ntotal, sizeof(float) * (size_t)h->ntotal * (size_t)(n - s));
        h->rbase = s;
    }
    nat_view(h, h->rows + (size_t)(s - h->rbase) * h->ntotal, n - s, r);
    return 0;
}

/* the stream's rows for words[s..n): the native state, or one pass over the window */
static int run_stream(arctag_t *h, int s, int n, const unsigned char *us, run_t *r) {
    if (h->backend == ARCTAG_BACKEND_NATIVE) return nat_window(h, s, n, r);
    return run_model(h, h->words, n, s, n, us, r);
}

/* probability of read-out q, row (window-relative) r, column c */
static float P(const arctag_t *h, const run_t *r, int q, int row, int c) {
    int oi = h->qo[q];
    return r->o[oi][(size_t)row * r->st[oi] + c];
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
        int c = idx[i]; float p = P(h, r, q, row, c); double th = h->thr[q][c];
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
    cJSON_AddItemToObject(ro, "valence", top(h, r, Q_UTT, row, "affect.valence.", 3, NULL, NULL, NULL));
    cJSON_AddItemToObject(ro, "intensity", top(h, r, Q_UTT, row, "affect.intensity.", 3, NULL, NULL, NULL));
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

/* numpy's float32 pairwise summation (np.add.reduce on a contiguous 1-D float32 array) */
static float pairwise_sum_f32(const float *a, int n) {
    if (n < 8) { float res = 0.0f; for (int i = 0; i < n; i++) res += a[i]; return res; }
    if (n <= 128) {
        float r[8], res; int i;
        for (int j = 0; j < 8; j++) r[j] = a[j];
        for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += a[i + j];
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    int n2 = n / 2; n2 -= n2 % 8;
    return pairwise_sum_f32(a, n2) + pairwise_sum_f32(a + n2, n - n2);
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
            ON[k * nin + c] = (double)P(h, r, Q_IN, row, c) >= (double)h->thr[Q_IN][c] && pf->valid[Q_IN][c];
            int bo = m->beg_of_in[c];
            if (bo >= 0) BEG[k * nin + c] = (double)P(h, r, Q_BEG, row, bo) >= (double)h->thr[Q_BEG][bo];
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
                    span_t s = {c, w0, w1, mx, h->thr[Q_IN][c], 0, 0, 0, 0, 0}; sp_push(&res, s);
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
                float M[64]; int cnt = 0;                      /* numpy float32 mean over rows (= reference) */
                for (int t = 0; t < ncols; t++) M[t] = 0.0f;
                for (int w = ga; w < gb; w++) {
                    int row = w - off; if (row < r0 || row >= r1) continue;
                    cnt++; for (int t = 0; t < ncols; t++) M[t] += P(h, r, Q_IN, row, cols[t]);
                }
                for (int t = 0; t < ncols; t++) M[t] /= (float)(cnt > 0 ? cnt : 1);
                int b1 = 0; for (int t = 1; t < ncols; t++) if (M[t] > M[b1]) b1 = t;
                int b2 = -1; for (int t = 0; t < ncols; t++) if (t != b1 && (b2 < 0 || M[t] > M[b2])) b2 = t;
                double mx = 0; int first = 1;
                for (int w = ga; w < gb; w++) { int row = w - off; if (row < r0 || row >= r1) continue; double p = P(h, r, Q_IN, row, cols[b1]); if (first || p > mx) mx = p; first = 0; }
                span_t s = {cols[b1], ga, gb, mx, h->thr[Q_IN][cols[b1]], 0, b2 >= 0, b2 >= 0 ? cols[b2] : 0, b2 >= 0 ? M[b2] : 0, 0};
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
        int nt = r->st[it], sb = r->st[ib];
        int mem[64], nm = 0;
        for (int t = 0; t < m->fam[f].ntags && nm < 64; t++) {
            int c = find_str(m->cols[Q_IN], nin, m->fam[f].tags[t]);
            if (c >= 0 && headcol[c] && pf->valid[Q_IN][c]) mem[nm++] = t;
        }
        double th = h->head[f];
        int *lab = (int *)calloc((size_t)W, sizeof(int));
        for (int k = 0; k < W; k++) {
            const float *pb = r->o[ib] + (size_t)(r0 + k) * sb;
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
                float sc[64], pm[64]; double mx = 0;          /* float32 sums (= reference) */
                for (int t = 0; t < nm; t++) { sc[t] = 0.0f; pm[t] = 0.0f; }
                for (int x = k; x < e; x++) {
                    const float *pt = r->o[it] + (size_t)(r0 + x) * nt;
                    for (int t = 0; t < nm; t++) {
                        float v = pt[mem[t]]; if (v < 1e-9f) v = 1e-9f; if (v > 1.0f) v = 1.0f;
                        sc[t] += logf(v); pm[t] += pt[mem[t]];
                    }
                    float pin = 1.0f - r->o[ib][(size_t)(r0 + x) * sb];
                    if (x == k || pin > mx) mx = pin;
                }
                int b1 = 0; for (int t = 1; t < nm; t++) if (sc[t] > sc[b1]) b1 = t;
                int b2 = -1; for (int t = 0; t < nm; t++) if (t != b1 && (b2 < 0 || sc[t] > sc[b2])) b2 = t;
                int c = find_str(m->cols[Q_IN], nin, m->fam[f].tags[mem[b1]]);
                span_t s = {c, w0, w1, mx, th, 1, b2 >= 0, 0, 0, 0};
                if (b2 >= 0) { s.alt_c = find_str(m->cols[Q_IN], nin, m->fam[f].tags[mem[b2]]); s.alt_p = pm[b2] / (float)(e - k); }
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
            int ra = s->a - off, rb = s->b - off;
            if (ra < 0) ra = 0;
            if (rb > r->W) rb = r->W;
            int cnt = rb > ra ? rb - ra : 0;
            float *vv = (float *)malloc(sizeof(float) * (size_t)(cnt > 0 ? cnt : 1));
            for (int k = 0; vv && k < cnt; k++) vv[k] = r->o[vm][(size_t)(ra + k) * r->st[vm]];
            float mean = cnt && vv ? pairwise_sum_f32(vv, cnt) / (float)cnt : 0.0f;    /* numpy float32 mean */
            free(vv);
            int rl = s->b - 1 - off;
            double v = (double)mean + (rl >= 0 && rl < r->W ? r->o[ve][(size_t)rl * r->st[ve]] : 0);
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
                if (!((double)p >= (double)h->thr[q][c] && pf->valid[q][c]) || h->m.ev_skip[q][c]) continue;
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
                cJSON_AddItemToObject(o, "thr", jnum(h->thr[q][c]));
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

/* ------------------------------------------------------------------ output sections, filters, redaction */
static int span_section(const char *t) {
    if (starts_with(t, "safety.toxicity.")) return E_TOX;
    if (starts_with(t, "safety.pii.")) return E_PII;
    if (starts_with(t, "safety.profanity.")) return E_PROF;
    if (starts_with(t, "entity.number.") || starts_with(t, "entity.numeral")) return E_NUMBER;
    if (starts_with(t, "entity.mention.")) return E_ENTITY;
    return -1;
}

static int rcat_of(const char *t) {
    static const char *tags[] = {"safety.pii.phone_number", "safety.pii.email", "safety.pii.street_address", "safety.pii.card_account_number",
                                 "safety.pii.gov_id", "safety.pii.date_of_birth", "safety.pii.private_name", "safety.pii.url_handle",
                                 "safety.pii.public_contact", "safety.profanity.mild", "safety.profanity.strong",
                                 "safety.toxicity.insult_harassment", "safety.toxicity.hate_identity", "safety.toxicity.threat_violence",
                                 "safety.toxicity.sexual_explicit", "safety.toxicity.self_harm"};
    static const int cat[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 10, 10};
    for (int i = 0; i < (int)(sizeof cat / sizeof cat[0]); i++) if (!strcmp(t, tags[i])) return cat[i];
    return -1;
}

static int keep_tag(const arctag_t *h, const char *t) {
    if (!h->ntagpre) return 1;
    for (int i = 0; i < h->ntagpre; i++) if (starts_with(t, h->tagpre[i])) return 1;
    return 0;
}

static const char *S_(cJSON *o, const char *k) { cJSON *x = cJSON_GetObjectItem(o, k); return cJSON_IsString(x) ? x->valuestring : ""; }
static int I_(cJSON *o, const char *k) { cJSON *x = cJSON_GetObjectItem(o, k); return cJSON_IsNumber(x) ? x->valueint : 0; }

/* stable sort of a cJSON array of spans by (start, -end) or, with by_type, (start, -end, type) */
static cJSON **span_vec(cJSON *arr, int *n, int by_type) {
    *n = cJSON_GetArraySize(arr);
    cJSON **v = (cJSON **)malloc(sizeof(cJSON *) * (size_t)(*n + 1)); int k = 0;
    for (cJSON *x = arr ? arr->child : NULL; x; x = x->next) v[k++] = x;
    for (int i = 1; i < k; i++) {
        cJSON *x = v[i]; int j = i - 1;
        while (j >= 0) {
            int a1 = I_(v[j], "start"), a2 = I_(x, "start"), b1 = I_(v[j], "end"), b2 = I_(x, "end"), gt;
            if (a1 != a2) gt = a1 > a2; else if (b1 != b2) gt = b1 < b2; else gt = by_type && strcmp(S_(v[j], "type"), S_(x, "type")) > 0;
            if (!gt) break;
            v[j + 1] = v[j]; j--;
        }
        v[j + 1] = x;
    }
    return v;
}

/* redacted text of words[a, b): words of masked categories become [PHONE], [NAME], ... (one placeholder per span run) */
static char *redact_text(const arctag_t *h, const word_t *ws, int a, int b, cJSON *spans) {
    int W = b - a > 0 ? b - a : 0, n;
    int *lc = (int *)malloc(sizeof(int) * (size_t)(W + 1)), *ls = (int *)malloc(sizeof(int) * (size_t)(W + 1));
    for (int i = 0; i < W; i++) lc[i] = -1;
    cJSON **v = span_vec(spans, &n, 1);
    for (int k = 0; k < n; k++) {
        int c = rcat_of(S_(v[k], "type"));
        if (c < 0 || !h->rpol[c]) continue;
        int s0 = I_(v[k], "start"), e0 = I_(v[k], "end");
        for (int i = s0 > a ? s0 : a; i < e0 && i < b; i++) if (lc[i - a] < 0) { lc[i - a] = c; ls[i - a] = s0; }
    }
    free(v);
    size_t L = 1; for (int i = a; i < b; i++) L += strlen(ws[i].w) + 12;
    char *out = (char *)malloc(L); size_t k = 0; out[0] = 0;
    for (int i = 0; i < W;) {
        if (k) out[k++] = ' ';
        if (lc[i] < 0) { size_t m = strlen(ws[a + i].w); memcpy(out + k, ws[a + i].w, m); k += m; i++; }
        else {
            int j = i; while (j < W && lc[j] == lc[i] && ls[j] == ls[i]) j++;
            k += (size_t)sprintf(out + k, "[%s]", RLAB[lc[i]]); i = j;
        }
    }
    out[k] = 0;
    free(lc); free(ls);
    return out;
}

static int filtering(const arctag_t *h) { return !h->all_emit || h->ntagpre; }

static void filter_list(const arctag_t *h, cJSON *arr, const char *key) {
    for (cJSON *x = arr ? arr->child : NULL; x;) { cJSON *nx = x->next; if (!keep_tag(h, S_(x, key))) cJSON_Delete(cJSON_DetachItemViaPointer(arr, x)); x = nx; }
}

static void filter_utt(const arctag_t *h, cJSON *u) {
    static const int ek[] = {E_INTENT, E_ACT, E_SAFETY, E_EMOTION, E_AROUSAL, E_VALENCE, E_INTENSITY, E_ROLE, E_ITEM, E_STATE};
    for (int i = 0; i < (int)(sizeof ek / sizeof ek[0]); i++) {
        const char *k = EMIT_N[ek[i]]; cJSON *x = cJSON_GetObjectItem(u, k);
        if (!x) continue;
        if (!h->emit[ek[i]]) { cJSON_DeleteItemFromObject(u, k); continue; }
        if (ek[i] != E_STATE) { filter_list(h, x, "tag"); continue; }
        for (cJSON *g = x->child; g;) {
            cJSON *ng = g->next; filter_list(h, g, "tag");
            if (!cJSON_GetArraySize(g)) cJSON_Delete(cJSON_DetachItemViaPointer(x, g));
            g = ng;
        }
    }
}

static void filter_result(const arctag_t *h, cJSON *res) {
    if (!filtering(h)) return;
    cJSON *sp = cJSON_GetObjectItem(res, "spans"), *ev = cJSON_GetObjectItem(res, "events"), *ut = cJSON_GetObjectItem(res, "utterances");
    for (cJSON *x = sp ? sp->child : NULL; x;) {
        cJSON *nx = x->next; int sec = span_section(S_(x, "type"));
        if (sec < 0 || !h->emit[sec] || !keep_tag(h, S_(x, "type"))) cJSON_Delete(cJSON_DetachItemViaPointer(sp, x));
        x = nx;
    }
    if (ev) { if (!h->emit[E_EVENTS]) while (cJSON_GetArraySize(ev)) cJSON_DeleteItemFromArray(ev, 0); else filter_list(h, ev, "type"); }
    for (cJSON *u = ut ? ut->child : NULL; u; u = u->next) filter_utt(h, u);
    cJSON *cur = cJSON_GetObjectItem(res, "current");
    if (cJSON_IsObject(cur)) filter_utt(h, cur);
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

static void set_model_path(arctag_t *h) {
    free(h->model_path);
    if (h->model_file && (strchr(h->model_file, '/') || strchr(h->model_file, '\\'))) h->model_path = xstrdup(h->model_file);
    else h->model_path = path_join(h->model_dir, h->model_file ? h->model_file : h->precision == ARCTAG_INT8 ? "arctag_int8.onnx" : "arctag.onnx");
}

void arctag_options_init(arctag_options *o) {
    if (!o) return;
    memset(o, 0, sizeof *o);
    o->size = sizeof *o; o->precision = ARCTAG_FP32; o->threads = 0; o->model_file = NULL; o->backend = ARCTAG_BACKEND_DEFAULT;
}

static const char *backend_name(int b) { return b == ARCTAG_BACKEND_ONNXRUNTIME ? "onnxruntime" : "native"; }
/* -1: unknown name */
static int backend_of(const char *v) {
    if (!strcmp(v, "native")) return ARCTAG_BACKEND_NATIVE;
    if (!strcmp(v, "onnxruntime") || !strcmp(v, "ort")) return ARCTAG_BACKEND_ONNXRUNTIME;
    if (!strcmp(v, "default") || !*v) return ARCTAG_BACKEND_NATIVE;
    return -1;
}

static void close_native(arctag_t *h) {
    nat_free(h->one); nat_free(h->st); nat_close(h->nm);
    h->one = h->st = NULL; h->nm = NULL;
    free(h->rows); h->rows = NULL; h->rcap = 0;
    stream_invalidate(h);
}

/* load the model file into the native backend and check it against arctag_meta.json */
static int open_native(arctag_t *h) {
    char e[512];
    nat_model *nm = nat_open(h->model_path, e, sizeof e);
    if (!nm) { set_err(h, "%s", e); return -1; }
    int off[NAT_NOUT], o = 0, noff[MAX_OUT], ndim[MAX_OUT];
    for (int k = 0; k < NAT_NOUT; k++) { off[k] = o; o += nat_out_dim(nm, k); }
    for (int i = 0; i < h->nout; i++) {
        int k = -1;
        for (int j = 0; j < NAT_NOUT; j++) if (!strcmp(h->out_names[i], nat_out_name(j))) k = j;
        if (k < 0) { set_err(h, "%s: no output %s", h->model_path, h->out_names[i]); nat_close(nm); return -1; }
        noff[i] = off[k]; ndim[i] = nat_out_dim(nm, k);
    }
    for (int q = 0; q < NQ; q++) if (h->m.nc[q] != ndim[h->qo[q]]) {
        set_err(h, "arctag_meta.json has %d columns for %s, %s has %d: they are not from the same model", h->m.nc[q], QN[q], h->model_path, ndim[h->qo[q]]);
        nat_close(nm); return -1;
    }
    for (int p = 0; p < h->m.nprof; p++) if (h->m.prof[p].dom < 0 || h->m.prof[p].dom >= nat_num_domains(nm)) {
        set_err(h, "arctag_meta.json: profile %s has domain %d, %s has %d", h->m.prof[p].name, h->m.prof[p].dom, h->model_path, nat_num_domains(nm));
        nat_close(nm); return -1;
    }
    for (int f = 0; f < h->m.nfam; f++) {
        int it = out_index(h, h->m.fam[f].out_t), ib = out_index(h, h->m.fam[f].out_b);
        if (it < 0 || ib < 0 || h->m.fam[f].ntags > ndim[it] || ndim[ib] != 3) {
            set_err(h, "arctag_meta.json: span family %s does not match the outputs of %s", h->m.fam[f].name, h->model_path);
            nat_close(nm); return -1;
        }
    }
    close_native(h);
    h->nm = nm; h->ntotal = o;
    memcpy(h->noff, noff, sizeof noff); memcpy(h->ndim, ndim, sizeof ndim);
    return 0;
}

/* (re)load the model for the current backend, precision and file; the old one stays in use if this fails */
static int open_session(arctag_t *h) {
    if (h->backend == ARCTAG_BACKEND_ONNXRUNTIME) {
#ifdef ARCTAG_WITH_ORT
        if (open_ort(h)) return -1;
        close_native(h);
        return 0;
#else
        set_err(h, "this build of arctag has no ONNX Runtime backend (configure with -DARCTAG_WITH_ORT=ON)");
        return -1;
#endif
    }
    if (open_native(h)) return -1;
#ifdef ARCTAG_WITH_ORT
    close_ort(h);
#endif
    return 0;
}

arctag_t *arctag_create_ex(const char *model_dir, const arctag_options *opt, char *err, size_t errlen) {
    arctag_t *h = (arctag_t *)calloc(1, sizeof(arctag_t));
    if (!h) { if (err && errlen) snprintf(err, errlen, "out of memory"); return NULL; }
    char *p;
    if (!model_dir) model_dir = "model";
    p = path_join(model_dir, "arctag_meta.json");
    if (load_meta(h, p)) { free(p); goto fail; }
    free(p);
    p = path_join(model_dir, "tokenizer.json");
    h->tok = arctok_load(p, h->err, sizeof h->err); free(p);
    if (!h->tok) goto fail;
    h->base_prof = h->m.default_prof; h->asr = h->m.default_asr; h->max_ctx = h->m.max_ctx; h->stream_ctx = 256;
    h->fine_auto = 1; h->fine_chk = -1; h->all_emit = 1;
    for (int e = 0; e < N_EMIT; e++) h->emit[e] = 1;
    use_profile(h, h->base_prof);
    { const char *t = getenv("ARCTAG_THREADS"); h->threads = t ? atoi(t) : 0; }
    { const char *t = getenv("ARCTAG_PRECISION"); h->precision = (t && !strcmp(t, "int8")) ? ARCTAG_INT8 : ARCTAG_FP32; }
    if (opt && opt->size >= offsetof(arctag_options, precision) + sizeof opt->precision) h->precision = opt->precision == ARCTAG_INT8 ? ARCTAG_INT8 : ARCTAG_FP32;
    if (opt && opt->size >= offsetof(arctag_options, threads) + sizeof opt->threads && opt->threads > 0) h->threads = opt->threads;
    if (opt && opt->size >= offsetof(arctag_options, model_file) + sizeof opt->model_file && opt->model_file) h->model_file = xstrdup(opt->model_file);
    h->backend = ARCTAG_BACKEND_NATIVE; h->stream_state = 1;
    { const char *t = getenv("ARCTAG_BACKEND"); int b = t ? backend_of(t) : -1; if (b > 0) h->backend = b; }
    if (opt && opt->size >= offsetof(arctag_options, backend) + sizeof opt->backend && opt->backend != ARCTAG_BACKEND_DEFAULT) {
        if (opt->backend != ARCTAG_BACKEND_NATIVE && opt->backend != ARCTAG_BACKEND_ONNXRUNTIME) { set_err(h, "unknown backend %d", (int)opt->backend); goto fail; }
        h->backend = opt->backend;
    }
    for (int q = 0; q < NQ; q++) { h->qo[q] = h->nout; h->out_names[h->nout++] = xstrdup(QN[q]); }
    for (int f = 0; f < h->m.nfam; f++) { h->out_names[h->nout++] = xstrdup(h->m.fam[f].out_t); h->out_names[h->nout++] = xstrdup(h->m.fam[f].out_b); }
    h->out_names[h->nout++] = xstrdup("ver_m"); h->out_names[h->nout++] = xstrdup("ver_e");
    h->model_dir = xstrdup(model_dir);
    set_model_path(h);
    if (open_session(h)) goto fail;
    h->err[0] = 0;
    return h;
fail:
    if (err && errlen) snprintf(err, errlen, "%s", h->err[0] ? h->err : "arctag_create failed");
    arctag_destroy(h);
    return NULL;
}

arctag_t *arctag_create(const char *model_dir, char *err, size_t errlen) { return arctag_create_ex(model_dir, NULL, err, errlen); }

void arctag_reset(arctag_t *h) {
    if (!h) return;
    for (int i = 0; i < h->n; i++) word_free(&h->words[i]);
    h->n = 0; h->nb = 0; h->np = 0; h->nk = 0; h->lo = 0;
    h->fine_chk = -1;
    stream_invalidate(h);
    if (h->m.prof && h->prof != h->base_prof) use_profile(h, h->base_prof);
}

void arctag_destroy(arctag_t *h) {
    if (!h) return;
    arctag_reset(h);
    free(h->words); free(h->bounds); free(h->pend); free(h->keys);
#ifdef ARCTAG_WITH_ORT
    close_ort(h);
#endif
    close_native(h);
    for (int i = 0; i < h->nout; i++) free(h->out_names[i]);
    for (int q = 0; q < NQ; q++) free(h->thr[q]);
    for (int i = 0; i < h->nov; i++) free(h->ov[i].pat);
    free(h->ov);
    for (int i = 0; i < h->ntagpre; i++) free(h->tagpre[i]);
    free(h->tagpre);
    if (h->m.prof) free_meta(&h->m);
    arctok_free(h->tok);
    free(h->model_path); free(h->model_dir); free(h->model_file);
    free(h);
}

static int q_index(const char *name) { for (int q = 0; q < NQ; q++) if (!strcmp(QN[q], name)) return q; return -1; }

static int set_threshold(arctag_t *h, const char *pattern, const char *value) {
    char pat[256]; int q = -1;
    snprintf(pat, sizeof pat, "%s", pattern);
    if (!strncmp(pat, "head:", 5)) { q = -2; memmove(pat, pat + 5, strlen(pat + 5) + 1); }
    else { char *at = strrchr(pat, '@'); if (at) { *at = 0; q = q_index(at + 1); if (q < 0) { set_err(h, "thr: unknown read-out '%s'", at + 1); return -1; } } }
    if (!*pat) { set_err(h, "thr: empty pattern"); return -1; }
    if (!value || !*value) {                                         /* remove the overrides of this pattern */
        int k = 0;
        for (int i = 0; i < h->nov; i++) { if (!strcmp(h->ov[i].pat, pat) && h->ov[i].q == q) free(h->ov[i].pat); else h->ov[k++] = h->ov[i]; }
        h->nov = k; use_profile(h, h->prof); return 0;
    }
    char *end; double v = strtod(value, &end);
    if (end == value || *end || v < 0 || v > 1.01) { set_err(h, "thr: value must be a number in 0..1.01"); return -1; }
    h->ov = (thr_ov_t *)realloc(h->ov, sizeof(thr_ov_t) * (size_t)(h->nov + 1));
    h->ov[h->nov].pat = xstrdup(pat); h->ov[h->nov].q = q; h->ov[h->nov].v = v; h->nov++;
    use_profile(h, h->prof);
    return 0;
}

static void split_list(const char *v, char ***out, int *n) {      /* comma / space separated items */
    *n = 0; *out = NULL;
    while (*v) {
        while (*v == ',' || *v == ' ') v++;
        const char *e = v; while (*e && *e != ',' && *e != ' ') e++;
        if (e > v) {
            *out = (char **)realloc(*out, sizeof(char *) * (size_t)(*n + 1));
            char *t = (char *)malloc((size_t)(e - v) + 1); memcpy(t, v, (size_t)(e - v)); t[e - v] = 0; (*out)[(*n)++] = t;
        }
        v = e;
    }
}

static int set_redact(arctag_t *h, const char *v) {
    if (!strcmp(v, "off") || !*v) { h->redact = 0; memset(h->rpol, 0, sizeof h->rpol); return 0; }
    if (!strcmp(v, "mask") || !strcmp(v, "on")) { h->redact = 1; for (int c = 0; c < N_RCAT; c++) h->rpol[c] = strcmp(RCAT[c], "contact") != 0; return 0; }
    unsigned char pol[N_RCAT] = {0};
    char **it; int n; split_list(v, &it, &n);
    int ok = 1;
    for (int i = 0; i < n; i++) {
        char *eq = strchr(it[i], '=');
        int c = -1;
        if (eq) { *eq = 0; for (int k = 0; k < N_RCAT; k++) if (!strcmp(RCAT[k], it[i])) c = k; }
        if (c < 0 || (strcmp(eq + 1, "mask") && strcmp(eq + 1, "tag") && strcmp(eq + 1, "off"))) { ok = 0; set_err(h, "redact: bad policy item (category=mask|tag|off)"); }
        else pol[c] = !strcmp(eq + 1, "mask");
    }
    for (int i = 0; i < n; i++) free(it[i]);
    free(it);
    if (!ok) return -1;
    h->redact = 1; memcpy(h->rpol, pol, sizeof pol);
    return 0;
}

static int set_emit(arctag_t *h, const char *v) {
    unsigned char em[N_EMIT] = {0};
    if (!*v || !strcmp(v, "all")) { for (int e = 0; e < N_EMIT; e++) h->emit[e] = 1; h->all_emit = 1; return 0; }
    char **it; int n; split_list(v, &it, &n);
    int ok = 1;
    for (int i = 0; i < n; i++) {
        int e = -1; for (int k = 0; k < N_EMIT; k++) if (!strcmp(EMIT_N[k], it[i])) e = k;
        if (e < 0) { ok = 0; set_err(h, "emit: unknown section '%s'", it[i]); } else em[e] = 1;
        free(it[i]);
    }
    free(it);
    if (!ok) return -1;
    memcpy(h->emit, em, sizeof em); h->all_emit = 1;
    for (int e = 0; e < N_EMIT; e++) if (!em[e]) h->all_emit = 0;
    return 0;
}

int arctag_set_option(arctag_t *h, const char *key, const char *value) {
    if (!h || !key || !value) return -1;
    if (!strcmp(key, "domain")) {
        int p = -1;
        for (int i = 0; i < h->m.nprof; i++) if (!strcmp(h->m.prof[i].name, value)) p = i;
        for (int i = 0; i < h->m.nalias && p < 0; i++) if (!strcmp(h->m.alias_k[i], value)) p = h->m.alias_v[i];
        if (p < 0) { set_err(h, "unknown domain '%s'", value); return -1; }
        h->base_prof = p; use_profile(h, p); arctag_reset(h); return 0;
    }
    if (!strcmp(key, "asr")) {
        int a = find_str(h->m.asr, h->m.nasr, !strcmp(value, "gold") ? "human" : value);   /* "gold": alias of "human" */
        if (a < 0) { set_err(h, "unknown asr source '%s'", value); return -1; }
        h->asr = a; arctag_reset(h); return 0;
    }
    if (!strcmp(key, "precision")) {
        int pr = !strcmp(value, "int8") ? ARCTAG_INT8 : !strcmp(value, "fp32") ? ARCTAG_FP32 : -1;
        if (pr < 0) { set_err(h, "precision must be fp32 or int8"); return -1; }
        if (pr == h->precision && !h->model_file) return 0;
        int old = h->precision; char *oldf = h->model_file;
        h->precision = pr; h->model_file = NULL; set_model_path(h);
        if (open_session(h)) { h->precision = old; h->model_file = oldf; set_model_path(h); return -1; }
        free(oldf);
        return 0;
    }
    if (!strcmp(key, "threads")) {
        int t = atoi(value);
        if (t < 0) { set_err(h, "threads must be >= 0"); return -1; }
        if (t == h->threads || h->backend == ARCTAG_BACKEND_NATIVE) { h->threads = t; return 0; }   /* native: one thread */
        int old = h->threads; h->threads = t;
        if (open_session(h)) { h->threads = old; return -1; }
        return 0;
    }
    if (!strcmp(key, "backend")) {
        int b = backend_of(value);
        if (b < 0) { set_err(h, "backend must be native or onnxruntime"); return -1; }
        if (b == h->backend) return 0;
        int old = h->backend; h->backend = b;
        if (open_session(h)) { h->backend = old; return -1; }
        return 0;
    }
    if (!strcmp(key, "stream_state")) {
        if (!strcmp(value, "on") || !strcmp(value, "1") || !strcmp(value, "true")) h->stream_state = 1;
        else if (!strcmp(value, "off") || !strcmp(value, "0") || !strcmp(value, "false")) h->stream_state = 0;
        else { set_err(h, "stream_state must be on or off"); return -1; }
        stream_invalidate(h);
        return 0;
    }
    if (!strcmp(key, "max_ctx")) { int v = atoi(value); if (v < 16 || v > 2048) { set_err(h, "max_ctx must be 16..2048"); return -1; } h->max_ctx = v; return 0; }
    if (!strcmp(key, "stream_ctx")) { int v = atoi(value); if (v < 16 || v > 2048) { set_err(h, "stream_ctx must be 16..2048"); return -1; } if (v != h->stream_ctx) stream_invalidate(h); h->stream_ctx = v; return 0; }
    if (!strcmp(key, "fine_auto")) {
        if (!strcmp(value, "on") || !strcmp(value, "1") || !strcmp(value, "true")) h->fine_auto = 1;
        else if (!strcmp(value, "off") || !strcmp(value, "0") || !strcmp(value, "false")) h->fine_auto = 0;
        else { set_err(h, "fine_auto must be on or off"); return -1; }
        return 0;
    }
    if (!strcmp(key, "emit")) return set_emit(h, value);
    if (!strcmp(key, "tags")) {
        for (int i = 0; i < h->ntagpre; i++) free(h->tagpre[i]);
        free(h->tagpre); h->tagpre = NULL; h->ntagpre = 0;
        if (*value && strcmp(value, "all")) split_list(value, &h->tagpre, &h->ntagpre);
        return 0;
    }
    if (!strcmp(key, "redact")) return set_redact(h, value);
    if (!strncmp(key, "thr:", 4)) return set_threshold(h, key + 4, value);
    if (!strcmp(key, "thr_reset")) { for (int i = 0; i < h->nov; i++) free(h->ov[i].pat); h->nov = 0; use_profile(h, h->prof); return 0; }
    set_err(h, "unknown option '%s'", key);
    return -1;
}

/* ------------------------------------------------------------------ config files */
static const char *CFG_INFO[] = {"name", "description", "vertical", "mode", "model_dir", "notes", "use_case", "output", NULL};

static int is_info_key(const char *k) {
    if (k[0] == '_') return 1;
    for (int i = 0; CFG_INFO[i]; i++) if (!strcmp(CFG_INFO[i], k)) return 1;
    return 0;
}

static char *join_strs(cJSON *arr) {
    size_t L = 1; for (cJSON *x = arr ? arr->child : NULL; x; x = x->next) if (cJSON_IsString(x)) L += strlen(x->valuestring) + 1;
    char *s = (char *)malloc(L); s[0] = 0;
    for (cJSON *x = arr ? arr->child : NULL; x; x = x->next) if (cJSON_IsString(x)) { if (*s) strcat(s, ","); strcat(s, x->valuestring); }
    return s;
}

static int apply_config_obj(arctag_t *h, cJSON *j) {
    char b[64];
    for (cJSON *x = j->child; x; x = x->next) {
        const char *k = x->string; int rc = 0;
        if (is_info_key(k)) continue;
        if (!strcmp(k, "domain") || !strcmp(k, "asr") || !strcmp(k, "precision") || !strcmp(k, "backend")) {
            if (!cJSON_IsString(x)) { set_err(h, "config: %s must be a string", k); return -1; }
            rc = arctag_set_option(h, k, x->valuestring);
        } else if (!strcmp(k, "threads") || !strcmp(k, "max_ctx") || !strcmp(k, "stream_ctx")) {
            if (!cJSON_IsNumber(x)) { set_err(h, "config: %s must be a number", k); return -1; }
            snprintf(b, sizeof b, "%d", x->valueint); rc = arctag_set_option(h, k, b);
        } else if (!strcmp(k, "fine_auto") || !strcmp(k, "stream_state")) {
            rc = arctag_set_option(h, k, cJSON_IsTrue(x) ? "on" : "off");
        } else if (!strcmp(k, "emit") || !strcmp(k, "tags")) {
            if (!cJSON_IsArray(x)) { set_err(h, "config: %s must be an array of strings", k); return -1; }
            char *s = join_strs(x); rc = arctag_set_option(h, k, cJSON_GetArraySize(x) || !strcmp(k, "tags") ? s : "all"); free(s);
        } else if (!strcmp(k, "thresholds")) {
            if (!cJSON_IsObject(x)) { set_err(h, "config: thresholds must be an object {pattern: value}"); return -1; }
            arctag_set_option(h, "thr_reset", "");
            for (cJSON *t = x->child; t && !rc; t = t->next) {
                if (!cJSON_IsNumber(t)) { set_err(h, "config: threshold '%s' must be a number", t->string); return -1; }
                snprintf(b, sizeof b, "%.17g", t->valuedouble); rc = set_threshold(h, t->string, b);
            }
        } else if (!strcmp(k, "redact")) {
            if (cJSON_IsString(x)) rc = set_redact(h, x->valuestring);
            else if (cJSON_IsObject(x)) {
                const char *mode = S_(x, "mode"); cJSON *pol = cJSON_GetObjectItem(x, "policy");
                if (!*mode || !strcmp(mode, "off")) rc = set_redact(h, "off");
                else if (strcmp(mode, "mask")) { set_err(h, "config: redact.mode must be off or mask"); return -1; }
                else if (cJSON_IsObject(pol) && pol->child) {
                    size_t L = 1; for (cJSON *t = pol->child; t; t = t->next) L += strlen(t->string) + (cJSON_IsString(t) ? strlen(t->valuestring) : 0) + 2;
                    char *s = (char *)calloc(L, 1);
                    for (cJSON *t = pol->child; t; t = t->next) { if (*s) strcat(s, ","); strcat(s, t->string); strcat(s, "="); strcat(s, cJSON_IsString(t) ? t->valuestring : "?"); }
                    rc = set_redact(h, s); free(s);
                } else rc = set_redact(h, "mask");
            } else { set_err(h, "config: redact must be a string or an object"); return -1; }
        } else { set_err(h, "config: unknown key '%s'", k); return -1; }
        if (rc) return -1;
    }
    return 0;
}

int arctag_apply_config(arctag_t *h, const char *config_json) {
    if (!h || !config_json) return -1;
    cJSON *j = cJSON_Parse(config_json);
    if (!cJSON_IsObject(j)) { set_err(h, "config: expected a JSON object"); cJSON_Delete(j); return -1; }
    int rc = apply_config_obj(h, j);
    cJSON_Delete(j);
    return rc;
}

arctag_t *arctag_create_from_config(const char *model_dir, const char *config_path, char *err, size_t errlen) {
    char *txt = config_path ? read_file(config_path) : NULL;
    if (!txt) { if (err && errlen) snprintf(err, errlen, "cannot read config %s", config_path ? config_path : "(null)"); return NULL; }
    cJSON *j = cJSON_Parse(txt); free(txt);
    if (!cJSON_IsObject(j)) { if (err && errlen) snprintf(err, errlen, "config %s: invalid JSON", config_path); cJSON_Delete(j); return NULL; }
    arctag_options o; arctag_options_init(&o);
    const char *pr = S_(j, "precision");
    if (!strcmp(pr, "int8")) o.precision = ARCTAG_INT8;
    cJSON *th = cJSON_GetObjectItem(j, "threads"); if (cJSON_IsNumber(th)) o.threads = th->valueint;
    const char *be = S_(j, "backend");
    if (*be) { int b = backend_of(be); if (b < 0) { if (err && errlen) snprintf(err, errlen, "config %s: backend must be native or onnxruntime", config_path); cJSON_Delete(j); return NULL; } o.backend = (arctag_backend_t)b; }
    char *md = NULL;
    if (!model_dir) {
        const char *rel = S_(j, "model_dir");
        if (*rel && rel[0] != '/' && !(rel[0] && rel[1] == ':')) {               /* relative to the config file */
            const char *sl = strrchr(config_path, '/'), *bs = strrchr(config_path, '\\');
            if (bs > sl) sl = bs;
            if (sl) { size_t n = (size_t)(sl - config_path); char *d = (char *)malloc(n + 1); memcpy(d, config_path, n); d[n] = 0; md = path_join(d, rel); free(d); }
            else md = xstrdup(rel);
        } else md = xstrdup(*rel ? rel : "model");
        model_dir = md;
    }
    arctag_t *h = arctag_create_ex(model_dir, &o, err, errlen);
    free(md);
    if (h && apply_config_obj(h, j)) {
        if (err && errlen) snprintf(err, errlen, "%s", h->err);
        arctag_destroy(h); h = NULL;
    }
    cJSON_Delete(j);
    return h;
}

static char *print_free(cJSON *o) { char *s = cJSON_PrintUnformatted(o); cJSON_Delete(o); return s; }

char *arctag_info(arctag_t *h) {
    if (!h) return NULL;
    cJSON *o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "version", ARCTAG_VERSION);
    cJSON_AddStringToObject(o, "precision", h->precision == ARCTAG_INT8 ? "int8" : "fp32");
    cJSON_AddStringToObject(o, "backend", backend_name(h->backend));
    if (h->backend == ARCTAG_BACKEND_NATIVE && h->nm) {
        cJSON_AddStringToObject(o, "weights", nat_weights_kind(h->nm));
        cJSON_AddStringToObject(o, "kernels", nat_simd_name());
        cJSON_AddBoolToObject(o, "stream_state", h->stream_state);
    }
    cJSON_AddStringToObject(o, "model_file", h->model_path);
    cJSON_AddStringToObject(o, "profile", h->m.prof[h->prof].name);
    cJSON_AddStringToObject(o, "asr", h->m.asr[h->asr]);
    cJSON_AddNumberToObject(o, "threads", h->threads);
    cJSON_AddNumberToObject(o, "max_ctx", h->max_ctx);
    cJSON_AddNumberToObject(o, "stream_ctx", h->stream_ctx);
    cJSON_AddBoolToObject(o, "fine_auto", h->fine_auto);
    cJSON *em = cJSON_AddArrayToObject(o, "emit");
    for (int e = 0; e < N_EMIT; e++) if (h->emit[e]) cJSON_AddItemToArray(em, cJSON_CreateString(EMIT_N[e]));
    cJSON *tg = cJSON_AddArrayToObject(o, "tags");
    for (int i = 0; i < h->ntagpre; i++) cJSON_AddItemToArray(tg, cJSON_CreateString(h->tagpre[i]));
    if (h->redact) {
        cJSON *rp = cJSON_AddObjectToObject(o, "redact");
        for (int c = 0; c < N_RCAT; c++) cJSON_AddStringToObject(rp, RCAT[c], h->rpol[c] ? "mask" : "off");
    } else cJSON_AddStringToObject(o, "redact", "off");
    cJSON *ov = cJSON_AddObjectToObject(o, "thresholds");
    for (int i = 0; i < h->nov; i++) {
        char k[300];
        if (h->ov[i].q == -2) snprintf(k, sizeof k, "head:%s", h->ov[i].pat);
        else if (h->ov[i].q >= 0) snprintf(k, sizeof k, "%s@%s", h->ov[i].pat, QN[h->ov[i].q]);
        else snprintf(k, sizeof k, "%s", h->ov[i].pat);
        cJSON_AddNumberToObject(ov, k, h->ov[i].v);
    }
    return print_free(o);
}

/* speaker slot of word `last` (same slot rule as the features), counting from word `from` */
static int spk_slot_at(const word_t *ws, int from, int last) {
    const char *names[16]; int ns = 0, slot = 0;
    for (int k = from; k <= last; k++) if (ws[k].spk) {
        int z = -1; for (int y = 0; y < ns; y++) if (!strcmp(names[y], ws[k].spk)) z = y;
        if (z < 0) { if (ns < 16) names[ns] = ws[k].spk; z = ns < 16 ? ns++ : 15; }
        if (k == last) slot = z + 1 > 9 ? 9 : z + 1;
    }
    return slot;
}

static void add_readouts(cJSON *o, cJSON *ro) {
    for (cJSON *x = ro->child; x;) { cJSON *nx = x->next; cJSON_AddItemToObject(o, x->string, cJSON_DetachItemViaPointer(ro, x)); x = nx; }
    cJSON_Delete(ro);
}

static int has_cues(const arctag_t *h, int p) {
    for (int i = 0; i < h->m.ncues; i++) if (!strcmp(h->m.cues[i].ui, h->m.prof[p].ui)) return 1;
    return 0;
}

/* offline tagging of words[0..n) with utterance starts us */
static char *tag_words_impl(arctag_t *h, word_t *ws, int n, unsigned char *us) {
    const int base = h->prof;
    /* fine-domain switches (after each utterance, when n // fine_every changed) */
    int seg_at[MAX_SEG + 1], seg_p[MAX_SEG], ns = 1;
    seg_at[0] = 0; seg_p[0] = base;
    if (h->fine_auto && has_cues(h, base) && n > 0) {
        const char **wv = (const char **)malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) wv[i] = ws[i].w;
        int chk = -1, cur = base;
        for (int e = 1; e <= n; e++) {
            if (e < n && !us[e]) continue;
            if (e / h->m.fine_every == chk) continue;
            chk = e / h->m.fine_every;
            int p = fine_hint(h, h->m.prof[base].ui, wv, e);
            if (p >= 0 && p != cur && ns < MAX_SEG) { seg_at[ns] = e; seg_p[ns++] = p; cur = p; }
        }
        free(wv);
    }
    seg_at[ns] = n + 1;
    int cut[MAX_SEG + 1];
    cut[0] = -1; for (int k = 1; k < ns; k++) cut[k] = seg_at[k] - h->m.span_win > 0 ? seg_at[k] - h->m.span_win : 0;
    cut[ns] = n + 1;

    cJSON *res = cJSON_CreateObject();
    cJSON_AddStringToObject(res, "profile", h->m.prof[seg_p[ns - 1]].name);
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
        run_t rr[MAX_SEG]; int nr = 0;
        cJSON *sp = cJSON_CreateArray();
        for (int k = 0; k < ns && ok; k++) {
            use_profile(h, seg_p[k]);
            if (run_model(h, ws + c0, W, 0, W, u, &rr[k])) { ok = 0; break; }
            nr++;
            cJSON *sk = decode_spans(h, &rr[k], u, 0, W, c0, -1, ws);
            for (cJSON *x = sk->child; x;) {
                cJSON *nx = x->next; int e = I_(x, "end");
                if (cut[k] < e && e <= cut[k + 1]) cJSON_AddItemToArray(sp, cJSON_DetachItemViaPointer(sk, x));
                x = nx;
            }
            cJSON_Delete(sk);
        }
        if (ok) {
            int m; cJSON **v = span_vec(sp, &m, 0);
            for (int i = 0; i < m; i++) cJSON_AddItemToArray(spans, cJSON_DetachItemViaPointer(sp, v[i]));
            free(v);
            for (int k = 0; k < ns; k++) {
                int hi = (c1 < seg_at[k + 1] ? c1 : seg_at[k + 1]) - c0;
                if (hi <= 0) continue;
                use_profile(h, seg_p[k]);
                decode_events(h, &rr[k], 0, hi, c0, ws, events);
            }
            int a = 0;
            for (int i = 1; i <= W; i++) {
                if (i < W && !u[i]) continue;
                int k = 0; for (int z = 1; z < ns; z++) if (seg_at[z] < i + c0) k = z;
                use_profile(h, seg_p[k]);
                cJSON *o = cJSON_CreateObject();
                cJSON_AddNumberToObject(o, "start", a + c0); cJSON_AddNumberToObject(o, "end", i + c0);
                char *t = join_words(ws, a + c0, i + c0); cJSON_AddStringToObject(o, "text", t); free(t);
                add_readouts(o, readouts(h, &rr[k], i - 1, spk_slot_at(ws, c0, i + c0 - 1)));
                cJSON_AddItemToArray(utts, o);
                a = i;
            }
        }
        cJSON_Delete(sp);
        for (int k = 0; k < nr; k++) run_free(h, &rr[k]);
        free(u);
    }
    use_profile(h, base);
    free(h->keys); h->keys = saved_keys; h->nk = saved_nk; h->kcap = saved_cap;
    if (!ok) { cJSON_Delete(res); return NULL; }
    if (h->redact) { char *r = redact_text(h, ws, 0, n, spans); cJSON_AddStringToObject(res, "redacted_text", r); free(r); }
    filter_result(h, res);
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
    if (from < h->sdirty) h->sdirty = from;                         /* native stream: words from here on changed */
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
    cJSON *prof = cJSON_AddStringToObject(res, "profile", h->m.prof[h->prof].name);
    if (!n) {
        cJSON_AddArrayToObject(res, "spans"); cJSON_AddArrayToObject(res, "events"); cJSON_AddArrayToObject(res, "utterances");
        cJSON_AddNullToObject(res, "current");
        return print_free(res);
    }
    int s = n - h->stream_ctx > 0 ? n - h->stream_ctx : 0, W = n - s;
    unsigned char *us = (unsigned char *)calloc((size_t)W, 1);
    us[0] = 1;
    for (int b = 0; b < h->nb; b++) if (h->bounds[b] >= s && h->bounds[b] < n) us[h->bounds[b] - s] = 1;
    run_t r, r2;
    if (run_stream(h, s, n, us, &r)) { free(us); cJSON_Delete(res); return NULL; }
    /* utterances closed since the last poll: read with the profile they were spoken under */
    cJSON *utts = cJSON_CreateArray();
    for (int i = 0; i < h->np; i++) {
        int bi = h->pend[i], end = h->bounds[bi], start = bi ? h->bounds[bi - 1] : 0;
        if (end - 1 < s) continue;                                   /* fell out of the window: skipped */
        cJSON *o = cJSON_CreateObject();
        cJSON_AddNumberToObject(o, "start", start); cJSON_AddNumberToObject(o, "end", end);
        char *t = join_words(h->words, start, end); cJSON_AddStringToObject(o, "text", t); free(t);
        cJSON_AddBoolToObject(o, "final", 1);
        add_readouts(o, readouts(h, &r, end - 1 - s, spk_slot_at(h->words, 0, end - 1)));
        cJSON_AddItemToArray(utts, o);
    }
    h->np = 0;
    int lo = h->lo > s ? h->lo : s;
    cJSON *evs = cJSON_CreateArray();
    /* fine domain inside sports / court from the lexicon cues of the words so far (when n // fine_every changed), after
     * each update: the new words' events of the old profile stay, spans / events / the running
     * read-out are then re-decoded with the new profile */
    if (h->fine_auto && n / h->m.fine_every != h->fine_chk && has_cues(h, h->prof)) {
        h->fine_chk = n / h->m.fine_every;
        const char **wv = (const char **)malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) wv[i] = h->words[i].w;
        int p = fine_hint(h, h->m.prof[h->prof].ui, wv, n);
        free(wv);
        if (p >= 0 && p != h->prof) {
            decode_events(h, &r, lo - s, W, s, h->words, evs);
            use_profile(h, p);
            run_free(h, &r);
            if (run_stream(h, s, n, us, &r2)) { free(us); cJSON_Delete(utts); cJSON_Delete(evs); cJSON_Delete(res); return NULL; }
            r = r2; lo = s;
            cJSON_SetValuestring(prof, h->m.prof[h->prof].name);
        }
    }
    int d0 = a0 - 16 > s ? a0 - 16 : s;
    cJSON *spans = decode_spans(h, &r, us, d0 - s, W, s, a0, h->words);
    cJSON_AddItemToObject(res, "spans", spans);
    decode_events(h, &r, lo - s, W, s, h->words, evs);
    h->lo = n;
    cJSON_AddItemToObject(res, "events", evs);
    cJSON_AddItemToObject(res, "utterances", utts);
    int ustart = h->nb ? h->bounds[h->nb - 1] : 0;
    if (ustart < n) {
        cJSON *o = cJSON_CreateObject();
        cJSON_AddNumberToObject(o, "start", ustart); cJSON_AddNumberToObject(o, "end", n);
        cJSON_AddBoolToObject(o, "final", 0);
        add_readouts(o, readouts(h, &r, W - 1, spk_slot_at(h->words, 0, n - 1)));
        cJSON_AddItemToObject(res, "current", o);
    } else cJSON_AddNullToObject(res, "current");
    if (h->redact) { char *rt = redact_text(h, h->words, a0, n, spans); cJSON_AddStringToObject(res, "redacted_text", rt); free(rt); }
    run_free(h, &r); free(us);
    filter_result(h, res);
    return print_free(res);
}
