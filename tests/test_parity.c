/* C library == Python reference (tests/expected/reference*.json): spans / events / utterance read-outs / redacted text
 * identical (tags, word ranges, on/off, field presence); scores equal within TOL (--tol, default 1e-6: the reference and the
 * library run the same ONNX graph; the residual is ONNX Runtime version / thread-count float noise). Cases with a "config"
 * are run on a fresh handle with arctag_apply_config. --tol 0 = bit-equal scores (reference made with onnxruntime 1.20.1, the
 * version the library links). Plus streaming: pushing the same words utterance by utterance and
 * polling gives the same spans, events and utterance read-outs as arctag_tag_text. With --int8 the streaming differences are
 * counted but do not fail: dynamic quantisation scales activations per call, so a shorter streaming window gives slightly
 * different scores than the full-text pass (see docs/int8.md); the int8_agreement test bounds them instead.
 *   test_parity MODEL_DIR reference.json [--int8] [--tol X] [--threads N] */
#include "arctag.h"
#include "cJSON.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double TOL = 1e-6, STOL = 1e-4;        /* STOL: streaming vs offline (partial-input runs: float noise only) */
static double *tolp = &TOL;
static int fails = 0, ndiff = 0, soft_stream = 0, nsoft = 0;
static double maxd = 0;
static const char *ctx = "";

static void bad(const char *what, const char *detail) {
    if (soft_stream && !strcmp(ctx, "stream")) { nsoft++; return; }
    fails++; if (fails <= 50) fprintf(stderr, "[%s] %s: %s\n", ctx, what, detail ? detail : ""); }
static const char *S(cJSON *o, const char *k) { cJSON *x = cJSON_GetObjectItem(o, k); return cJSON_IsString(x) ? x->valuestring : ""; }
static double N(cJSON *o, const char *k) { cJSON *x = cJSON_GetObjectItem(o, k); return cJSON_IsNumber(x) ? x->valuedouble : NAN; }
static void num(const char *what, double a, double b) {
    if (isnan(a) && isnan(b)) return;
    double d = fabs(a - b);
    if (tolp == &TOL) { if (d > maxd) maxd = d; if (d > 0) ndiff++; }
    if (!(d <= *tolp)) { char m[128]; snprintf(m, sizeof m, "%.6f vs %.6f", a, b); bad(what, m); }
}
static void str(const char *what, const char *a, const char *b) {
    if (strcmp(a, b)) { char m[512]; snprintf(m, sizeof m, "'%.200s' vs '%.200s'", a, b); bad(what, m); }
}

static void cmp_list(const char *what, cJSON *a, cJSON *b, const char *name_key, const char *score_key) {
    if ((a == NULL) != (b == NULL)) { bad(what, "presence"); return; }
    int na = cJSON_GetArraySize(a), nb = cJSON_GetArraySize(b);
    if (na != nb) { char m[64]; snprintf(m, sizeof m, "%d vs %d items", na, nb); bad(what, m); return; }
    for (cJSON *x = a ? a->child : NULL, *y = b ? b->child : NULL; x && y; x = x->next, y = y->next) {
        str(what, S(x, name_key), S(y, name_key));
        if (score_key) num(what, N(x, score_key), N(y, score_key));
        num(what, N(x, "thr"), N(y, "thr"));
        cJSON *ox = cJSON_GetObjectItem(x, "on"), *oy = cJSON_GetObjectItem(y, "on");
        if ((ox == NULL) != (oy == NULL) || (ox && cJSON_IsTrue(ox) != cJSON_IsTrue(oy))) bad(what, "on differs");
        const char *keys[] = {"start", "end", "word"};
        for (int k = 0; k < 3; k++) {
            cJSON *p = cJSON_GetObjectItem(x, keys[k]), *q = cJSON_GetObjectItem(y, keys[k]);
            if (p && q && p->valueint != q->valueint) bad(what, keys[k]);
            if ((p == NULL) != (q == NULL)) bad(what, "field presence");
        }
        if (!strcmp(what, "spans")) {
            str("span src", S(x, "src"), S(y, "src"));
            str("span text", S(x, "text"), S(y, "text"));
            num("span ver", N(x, "ver"), N(y, "ver"));
            cJSON *ax = cJSON_GetObjectItem(x, "alt"), *ay = cJSON_GetObjectItem(y, "alt");
            if ((ax == NULL) != (ay == NULL)) bad("span alt", "presence");
            else if (ax) { str("span alt", S(ax, "tag"), S(ay, "tag")); num("span alt p", N(ax, "p"), N(ay, "p")); }
        }
    }
}

static const char *RO[] = {"intent", "act", "safety", "emotion", "arousal", "valence", "intensity", "role", "item"};
#define NRO 9

static void cmp_utts(cJSON *a, cJSON *b, int check_text) {
    int na = cJSON_GetArraySize(a), nb = cJSON_GetArraySize(b);
    if (na != nb) { bad("utterances", "count"); return; }
    for (cJSON *x = a->child, *y = b->child; x && y; x = x->next, y = y->next) {
        if (N(x, "start") != N(y, "start") || N(x, "end") != N(y, "end")) bad("utterance", "range");
        if (check_text) str("utterance text", S(x, "text"), S(y, "text"));
        for (int k = 0; k < NRO; k++) cmp_list(RO[k], cJSON_GetObjectItem(x, RO[k]), cJSON_GetObjectItem(y, RO[k]), "tag", "p");
        cJSON *sx = cJSON_GetObjectItem(x, "state"), *sy = cJSON_GetObjectItem(y, "state");
        if ((sx == NULL) != (sy == NULL)) { bad("state", "presence"); continue; }
        if (cJSON_GetArraySize(sx) != cJSON_GetArraySize(sy)) bad("state", "groups");
        for (cJSON *g = sx ? sx->child : NULL; g; g = g->next) cmp_list("state", g, cJSON_GetObjectItem(sy, g->string), "tag", "p");
    }
}

static char *slurp(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1); if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f); return b;
}

/* streaming: words pushed utterance by utterance (endpoint after each), poll after every push */
static void stream_check(arctag_t *h, cJSON *res) {
    cJSON *words = cJSON_GetObjectItem(res, "words"), *utts = cJSON_GetObjectItem(res, "utterances");
    cJSON *ev_all = cJSON_CreateArray(), *ut_all = cJSON_CreateArray(), *last = NULL;
    arctag_reset(h);
    for (cJSON *u = utts->child; u; u = u->next) {
        int a = (int)N(u, "start"), b = (int)N(u, "end");
        cJSON *req = cJSON_CreateObject(), *ws = cJSON_AddArrayToObject(req, "words");
        for (int i = a; i < b; i++) cJSON_AddItemToArray(ws, cJSON_CreateString(cJSON_GetArrayItem(words, i)->valuestring));
        cJSON_AddBoolToObject(req, "endpoint", 1);
        char *js = cJSON_PrintUnformatted(req); cJSON_Delete(req);
        if (arctag_push_words(h, js)) bad("stream push", arctag_last_error(h));
        free(js);
        char *out = arctag_poll(h);
        cJSON *p = out ? cJSON_Parse(out) : NULL; arctag_free_str(out);
        if (!p) { bad("stream poll", arctag_last_error(h)); return; }
        cJSON *x;
        while ((x = cJSON_DetachItemFromArray(cJSON_GetObjectItem(p, "events"), 0))) cJSON_AddItemToArray(ev_all, x);
        while ((x = cJSON_DetachItemFromArray(cJSON_GetObjectItem(p, "utterances"), 0))) cJSON_AddItemToArray(ut_all, x);
        cJSON_Delete(last); last = p;
    }
    const char *c0 = ctx; ctx = "stream"; tolp = &STOL;
    cmp_list("spans", cJSON_GetObjectItem(last, "spans"), cJSON_GetObjectItem(res, "spans"), "type", "score");
    cmp_list("events", ev_all, cJSON_GetObjectItem(res, "events"), "type", "score");
    cmp_utts(ut_all, utts, 1);
    str("stream profile", S(last, "profile"), S(res, "profile"));
    ctx = c0; tolp = &TOL;
    cJSON_Delete(last); cJSON_Delete(ev_all); cJSON_Delete(ut_all);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL_DIR reference.json [--int8] [--tol X] [--threads N]\n", argv[0]); return 2; }
    arctag_options o; arctag_options_init(&o);
    o.threads = 1;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--int8")) { o.precision = ARCTAG_INT8; soft_stream = 1; }
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc) TOL = atof(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) o.threads = atoi(argv[++i]);
    }
    char err[512];
    arctag_t *h = arctag_create_ex(argv[1], &o, err, sizeof err);
    if (!h) {   /* 77 = skipped: the int8 graph is optional in a model dir */
        fprintf(stderr, "arctag_create: %s\n", err);
        return o.precision == ARCTAG_INT8 && strstr(err, "arctag_int8.onnx") ? 77 : 1;
    }
    char *txt = slurp(argv[2]); if (!txt) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    cJSON *ref = cJSON_Parse(txt); free(txt);
    int n = 0, nspans = 0, nopt = 0;
    for (cJSON *c = cJSON_GetObjectItem(ref, "cases")->child; c; c = c->next, n++) {
        char name[64]; snprintf(name, sizeof name, "case %d (%s)", n, S(c, "domain")); ctx = name;
        cJSON *cfg = cJSON_GetObjectItem(c, "config");
        arctag_t *hc = h;
        if (cfg) {                                                    /* fresh handle + config */
            hc = arctag_create_ex(argv[1], &o, err, sizeof err);
            char *cs = cJSON_PrintUnformatted(cfg);
            if (!hc || arctag_apply_config(hc, cs)) { bad("config", hc ? arctag_last_error(hc) : err); free(cs); arctag_destroy(hc); continue; }
            free(cs); nopt++;
        }
        if (arctag_set_option(hc, "domain", S(c, "domain"))) { bad("domain", arctag_last_error(hc)); continue; }
        char *out = arctag_tag_text(hc, S(c, "text"));
        cJSON *got = out ? cJSON_Parse(out) : NULL; arctag_free_str(out);
        if (!got) { bad("tag_text", arctag_last_error(hc)); continue; }
        cJSON *want = cJSON_GetObjectItem(c, "result");
        str("profile", S(got, "profile"), S(want, "profile"));
        cmp_list("spans", cJSON_GetObjectItem(got, "spans"), cJSON_GetObjectItem(want, "spans"), "type", "score");
        cmp_list("events", cJSON_GetObjectItem(got, "events"), cJSON_GetObjectItem(want, "events"), "type", "score");
        cmp_utts(cJSON_GetObjectItem(got, "utterances"), cJSON_GetObjectItem(want, "utterances"), 1);
        cJSON *rg = cJSON_GetObjectItem(got, "redacted_text"), *rw = cJSON_GetObjectItem(want, "redacted_text");
        if ((rg == NULL) != (rw == NULL)) bad("redacted_text", "presence");
        else if (rg) str("redacted_text", rg->valuestring, rw->valuestring);
        nspans += cJSON_GetArraySize(cJSON_GetObjectItem(want, "spans"));
        stream_check(hc, got);
        cJSON_Delete(got);
        if (hc != h) arctag_destroy(hc);
    }
    printf("parity (%s): %d cases (%d with a config), %d reference spans, %d mismatches, max |score diff| %.2e (tol %.0e), %d scores not bit-equal (streaming tol %.0e)\n",
           o.precision == ARCTAG_INT8 ? "int8" : "fp32", n, nopt, nspans, fails, maxd, TOL, ndiff, STOL);
    if (soft_stream) printf("streaming vs offline (int8, not checked: per-call activation scales): %d differing fields\n", nsoft);
    cJSON_Delete(ref); arctag_destroy(h);
    return fails ? 1 : 0;
}
