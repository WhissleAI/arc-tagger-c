/* Native backend: streaming state, revisions, long sessions, both backends on the same input, and bad model files.
 *   test_native MODEL_DIR TEXT_FILE...
 * 1. streaming with the state kept (stream_state on) == streaming that re-runs the window (off), poll by poll, while the
 *    session fits in stream_ctx; with revisions (words replaced from an earlier index), endpoints and a speaker label
 *    that appears mid-stream;
 * 2. a long session (several times stream_ctx) runs with bounded memory and keeps giving output;
 * 3. if the library has the ONNX Runtime backend: both backends give the same decoded output for the texts (scores
 *    within --tol, default 5e-4: ONNX Runtime's float32 kernels differ from exact arithmetic by up to about 2e-4);
 * 4. truncated and corrupted copies of arctag.onnx fail to load with a message, or load and run, without crashing. */
#include "arctag.h"
#include "cJSON.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static double TOL = 5e-4, maxd = 0;
static void bad(const char *what, const char *detail) { fails++; if (fails <= 40) fprintf(stderr, "FAIL %s: %s\n", what, detail ? detail : ""); }

static char *slurp(const char *p, long *len) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f); if (len) *len = n; return b;
}

/* compare two JSON values: same structure and strings, numbers within tol */
static void cmp_json(const char *where, cJSON *a, cJSON *b, double tol) {
    if (!a || !b) { if (a != b) bad(where, "presence"); return; }
    if ((a->type & 0xFF) != (b->type & 0xFF)) { bad(where, "type"); return; }
    if (cJSON_IsNumber(a)) {
        double d = fabs(a->valuedouble - b->valuedouble); if (d > maxd) maxd = d;
        if (d > tol) { char m[128]; snprintf(m, sizeof m, "%.6f vs %.6f", a->valuedouble, b->valuedouble); bad(where, m); }
    } else if (cJSON_IsString(a)) { if (strcmp(a->valuestring, b->valuestring)) { char m[256]; snprintf(m, sizeof m, "'%.100s' vs '%.100s'", a->valuestring, b->valuestring); bad(where, m); } }
    else if (cJSON_IsArray(a) || cJSON_IsObject(a)) {
        if (cJSON_GetArraySize(a) != cJSON_GetArraySize(b)) { char m[64]; snprintf(m, sizeof m, "%d vs %d items", cJSON_GetArraySize(a), cJSON_GetArraySize(b)); bad(where, m); return; }
        for (cJSON *x = a->child, *y = b->child; x && y; x = x->next, y = y->next) {
            char w[256]; snprintf(w, sizeof w, "%s.%s", where, x->string ? x->string : "[]");
            if (x->string && (!y->string || strcmp(x->string, y->string))) { bad(w, "key"); continue; }
            cmp_json(w, x, y, tol);
        }
    } else if (cJSON_IsBool(a) && cJSON_IsTrue(a) != cJSON_IsTrue(b)) bad(where, "bool");
}

static arctag_t *open_h(const char *dir, arctag_backend_t be) {
    char err[512]; arctag_options o; arctag_options_init(&o); o.backend = be; o.threads = 1;
    arctag_t *h = arctag_create_ex(dir, &o, err, sizeof err);
    if (!h) fprintf(stderr, "create: %s\n", err);
    return h;
}

static int split_words(char *text, char **w, int cap) {
    int n = 0;
    for (char *t = strtok(text, " \t\r\n"); t && n < cap; t = strtok(NULL, " \t\r\n")) w[n++] = t;
    return n;
}

static void push_json(arctag_t *h, char **w, int a, int b, int from, int endpoint, const char *spk) {
    cJSON *req = cJSON_CreateObject(), *ws = cJSON_AddArrayToObject(req, "words");
    for (int i = a; i < b; i++) {
        if (spk) { cJSON *o = cJSON_CreateObject(); cJSON_AddStringToObject(o, "w", w[i]); cJSON_AddStringToObject(o, "spk", spk); cJSON_AddItemToArray(ws, o); }
        else cJSON_AddItemToArray(ws, cJSON_CreateString(w[i]));
    }
    if (from >= 0) cJSON_AddNumberToObject(req, "from", from);
    if (endpoint) cJSON_AddBoolToObject(req, "endpoint", 1);
    char *s = cJSON_PrintUnformatted(req); cJSON_Delete(req);
    if (arctag_push_words(h, s)) bad("push", arctag_last_error(h));
    free(s);
}
static cJSON *poll_j(arctag_t *h) { char *s = arctag_poll(h); cJSON *j = s ? cJSON_Parse(s) : NULL; if (!j) bad("poll", arctag_last_error(h)); arctag_free_str(s); return j; }

/* 1: the same pushes into a stateful and a re-running handle, poll by poll */
static void stream_equivalence(const char *dir, char **w, int n) {
    arctag_t *a = open_h(dir, ARCTAG_BACKEND_NATIVE), *b = open_h(dir, ARCTAG_BACKEND_NATIVE);
    if (!a || !b) { bad("stream", "create"); arctag_destroy(a); arctag_destroy(b); return; }
    arctag_set_option(b, "stream_state", "off");
    static char *alt[] = {"seven", "hundred", "dollars", "on", "Friday"};
    char *mix[2048]; int polls = 0;
    if (n > 200) n = 200;                          /* within stream_ctx (256) */
    for (int i = 0; i < n; i++) mix[i] = w[i];
    int i = 0;
    while (i < n) {
        int k = 1 + (i * 7) % 5; if (i + k > n) k = n - i;
        const char *spk = i > n / 2 ? (i % 3 ? "A" : "B") : NULL;   /* speaker labels appear mid-stream */
        int ep = strchr(".?!", w[i + k - 1][strlen(w[i + k - 1]) - 1]) != NULL;
        for (int v = 0; v < 2; v++) push_json(v ? b : a, mix, i, i + k, -1, ep, spk);
        i += k;
        if (i > 20 && i % 13 == 0) {                /* a revision: replace the last few words, then restore them */
            int from = i - 4, m = i - from;
            char *tmp[8]; for (int z = 0; z < m; z++) { tmp[z] = alt[z % 5]; }
            for (int v = 0; v < 2; v++) push_json(v ? b : a, tmp, 0, m, from, 0, spk);
            cJSON *pa = poll_j(a), *pb = poll_j(b); cmp_json("revised poll", pa, pb, 1e-4); cJSON_Delete(pa); cJSON_Delete(pb); polls++;
            for (int v = 0; v < 2; v++) push_json(v ? b : a, mix, from, i, from, 0, spk);
        }
        if (i > 40 && i % 29 == 0) {                /* a deep revision: beyond the newest snapshots */
            int from = i - 37;
            for (int v = 0; v < 2; v++) push_json(v ? b : a, mix, from, i, from, 0, spk);
        }
        cJSON *pa = poll_j(a), *pb = poll_j(b);
        cmp_json("poll", pa, pb, 1e-4);
        cJSON_Delete(pa); cJSON_Delete(pb); polls++;
    }
    printf("stream_state on == off: %d words, %d polls\n", n, polls);
    arctag_destroy(a); arctag_destroy(b);
}

/* 2: a long session */
static void long_session(const char *dir, char **w, int n) {
    arctag_t *a = open_h(dir, ARCTAG_BACKEND_NATIVE);
    if (!a) { bad("long", "create"); return; }
    arctag_set_option(a, "stream_ctx", "64");
    int total = 0, last_n = 0, ok = 1;
    for (int rep = 0; rep < 12 && ok; rep++)
        for (int i = 0; i < n; i += 3) {
            int k = i + 3 < n ? 3 : n - i;
            push_json(a, w, i, i + k, -1, 0, NULL); total += k;
            cJSON *p = poll_j(a); if (!p) { ok = 0; break; }
            last_n = cJSON_GetObjectItem(p, "n_words")->valueint;
            if (!cJSON_IsObject(cJSON_GetObjectItem(p, "current")) && !cJSON_IsNull(cJSON_GetObjectItem(p, "current"))) bad("long", "current");
            cJSON_Delete(p);
        }
    if (last_n != total) bad("long", "word count");
    printf("long session: %d words with stream_ctx 64\n", total);
    arctag_destroy(a);
}

/* 3: both backends, offline and streaming */
static void backends(const char *dir, const char *const *texts, int nt) {
    arctag_t *a = open_h(dir, ARCTAG_BACKEND_NATIVE), *b = open_h(dir, ARCTAG_BACKEND_ONNXRUNTIME);
    if (!b) { printf("backends: no onnxruntime backend in this build, skipped\n"); arctag_destroy(a); return; }
    if (!a) { bad("backends", "create"); arctag_destroy(b); return; }
    for (int t = 0; t < nt; t++) {
        char *x = arctag_tag_text(a, texts[t]), *y = arctag_tag_text(b, texts[t]);
        cJSON *ja = x ? cJSON_Parse(x) : NULL, *jb = y ? cJSON_Parse(y) : NULL;
        cmp_json("tag_text", ja, jb, TOL);
        cJSON_Delete(ja); cJSON_Delete(jb); arctag_free_str(x); arctag_free_str(y);
    }
    arctag_set_option(a, "max_ctx", "40"); arctag_set_option(b, "max_ctx", "40");     /* passes split into chunks */
    for (int t = 0; t < nt; t++) {
        char *x = arctag_tag_text(a, texts[t]), *y = arctag_tag_text(b, texts[t]);
        cJSON *ja = x ? cJSON_Parse(x) : NULL, *jb = y ? cJSON_Parse(y) : NULL;
        cmp_json("tag_text max_ctx 40", ja, jb, TOL);
        cJSON_Delete(ja); cJSON_Delete(jb); arctag_free_str(x); arctag_free_str(y);
    }
    printf("backends agree on %d texts (max |score diff| %.2e, tol %.0e)\n", nt, maxd, TOL);
    arctag_destroy(a); arctag_destroy(b);
}

/* 4: bad model files */
static void bad_files(const char *dir, const char *tmpdir) {
    char src[1024], dst[1024]; long len = 0;
    snprintf(src, sizeof src, "%s/arctag.onnx", dir);
    char *buf = slurp(src, &len);
    if (!buf) { bad("bad files", "cannot read arctag.onnx"); return; }
    snprintf(dst, sizeof dst, "%s/arctag_bad.onnx", tmpdir);
    unsigned s = 12345; int loaded = 0, refused = 0;
    for (int it = 0; it < 24; it++) {
        long n = len; char *b = (char *)malloc((size_t)len); memcpy(b, buf, (size_t)len);
        if (it < 8) n = (long)((double)len * (it + 1) / 9.0);           /* truncated */
        else for (int f = 0; f < 1 + it % 4; f++) {                       /* flipped bytes, most in the first 64 KB (structure) */
            s = s * 1103515245u + 12345u; long at = f ? (long)(s % (unsigned)len) : (long)(s % 65536u);
            b[at] ^= (char)(1 + (s >> 16) % 255);
        }
        FILE *f = fopen(dst, "wb"); if (!f) { bad("bad files", "cannot write"); free(b); break; }
        fwrite(b, 1, (size_t)n, f); fclose(f); free(b);
        char err[512]; arctag_options o; arctag_options_init(&o); o.backend = ARCTAG_BACKEND_NATIVE; o.model_file = dst;
        arctag_t *h = arctag_create_ex(dir, &o, err, sizeof err);
        if (!h) { refused++; if (!err[0]) bad("bad files", "no message"); continue; }
        loaded++;
        char *out = arctag_tag_text(h, "Call me at 555 123 4567 tomorrow, Maria.");
        arctag_free_str(out);
        arctag_destroy(h);
    }
    remove(dst);
    free(buf);
    printf("bad model files: %d refused with a message, %d loaded and ran\n", refused, loaded);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL_DIR TEXT_FILE... [--tol X] [--tmp DIR]\n", argv[0]); return 2; }
    const char *texts[64]; char *owned[128]; int nt = 0, no = 0; const char *tmp = ".";
    static char *w[100000]; int nw = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--tol") && i + 1 < argc) { TOL = atof(argv[++i]); continue; }
        if (!strcmp(argv[i], "--tmp") && i + 1 < argc) { tmp = argv[++i]; continue; }
        char *t = slurp(argv[i], NULL);
        if (!t) { fprintf(stderr, "cannot read %s\n", argv[i]); return 2; }
        char *c = (char *)malloc(strlen(t) + 1); memcpy(c, t, strlen(t) + 1);
        if (nt < 64) { texts[nt++] = t; owned[no++] = t; owned[no++] = c; } else { free(t); owned[no++] = c; }
        nw += split_words(c, w + nw, 100000 - nw);
    }
    stream_equivalence(argv[1], w, nw);
    long_session(argv[1], w, nw < 300 ? nw : 300);
    backends(argv[1], texts, nt);
    bad_files(argv[1], tmp);
    printf("test_native: %d failures\n", fails);
    for (int i = 0; i < no; i++) free(owned[i]);
    return fails ? 1 : 0;
}
