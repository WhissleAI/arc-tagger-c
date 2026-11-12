/* int8 vs fp32 on the reference texts (tests/expected/reference.json cases without a config): decoded spans (exact type +
 * word range), events (type + word) and utterance read-outs that are "on" must agree with F1 >= --min-f1 (default 0.90), and
 * the max |p| difference of the reported scores stays under --max-dp (default 0.25). Prints per-family numbers.
 * --backend B runs both on that backend; --model-file F takes the int8 model from F instead of arctag_int8.onnx.
 *   test_int8_agreement MODEL_DIR reference.json [--min-f1 X] [--max-dp X] [--backend native|onnxruntime] [--model-file F] */
#include "arctag.h"
#include "cJSON.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1); if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f); return b;
}

#define NG 8
static const char *GN[NG] = {"entity", "number", "pii", "profanity", "events", "intent", "safety", "other_readouts"};
static long cnt[NG][3];       /* fp32, int8, both */
static double maxdp = 0;

static int group_of_span(const char *fam) { return !strcmp(fam, "entity") ? 0 : !strcmp(fam, "number") ? 1 : !strcmp(fam, "pii") ? 2 : 3; }
static int group_of_ro(const char *k) { return !strcmp(k, "intent") ? 5 : !strcmp(k, "safety") ? 6 : 7; }

static int keys(cJSON *r, char (*k)[200], int *g, double *p, int cap) {
    int n = 0;
    for (cJSON *s = cJSON_GetObjectItem(r, "spans")->child; s && n < cap; s = s->next, n++) {
        snprintf(k[n], 200, "s %s %d %d", cJSON_GetStringValue(cJSON_GetObjectItem(s, "type")), cJSON_GetObjectItem(s, "start")->valueint, cJSON_GetObjectItem(s, "end")->valueint);
        g[n] = group_of_span(cJSON_GetStringValue(cJSON_GetObjectItem(s, "family"))); p[n] = cJSON_GetObjectItem(s, "score")->valuedouble;
    }
    for (cJSON *e = cJSON_GetObjectItem(r, "events")->child; e && n < cap; e = e->next, n++) {
        snprintf(k[n], 200, "e %s %d", cJSON_GetStringValue(cJSON_GetObjectItem(e, "type")), cJSON_GetObjectItem(e, "word")->valueint);
        g[n] = 4; p[n] = cJSON_GetObjectItem(e, "score")->valuedouble;
    }
    for (cJSON *u = cJSON_GetObjectItem(r, "utterances")->child; u; u = u->next)
        for (cJSON *a = u->child; a; a = a->next) {
            cJSON *lists[64]; int nl = 0;
            if (cJSON_IsArray(a)) lists[nl++] = a;
            else if (cJSON_IsObject(a)) for (cJSON *gg = a->child; gg && nl < 64; gg = gg->next) lists[nl++] = gg;
            for (int l = 0; l < nl; l++)
                for (cJSON *x = lists[l]->child; x && n < cap; x = x->next)
                    if (cJSON_IsTrue(cJSON_GetObjectItem(x, "on"))) {
                        snprintf(k[n], 200, "u %d %s %s", cJSON_GetObjectItem(u, "start")->valueint, a->string, cJSON_GetStringValue(cJSON_GetObjectItem(x, "tag")));
                        g[n] = group_of_ro(a->string); p[n] = cJSON_GetObjectItem(x, "p")->valuedouble; n++;
                    }
        }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL_DIR reference.json [--min-f1 X] [--max-dp X]\n", argv[0]); return 2; }
    double min_f1 = 0.90, max_dp = 0.25; const char *mfile = NULL; arctag_backend_t be = ARCTAG_BACKEND_DEFAULT;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--min-f1") && i + 1 < argc) min_f1 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-dp") && i + 1 < argc) max_dp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--model-file") && i + 1 < argc) mfile = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) { i++; be = !strcmp(argv[i], "native") ? ARCTAG_BACKEND_NATIVE : ARCTAG_BACKEND_ONNXRUNTIME; }
    }
    char err[512]; arctag_t *h[2];
    for (int v = 0; v < 2; v++) {
        arctag_options o; arctag_options_init(&o); o.precision = v ? ARCTAG_INT8 : ARCTAG_FP32; o.threads = 1; o.backend = be;
        if (v && mfile) o.model_file = mfile;
        h[v] = arctag_create_ex(argv[1], &o, err, sizeof err);
        if (!h[v]) { fprintf(stderr, "%s: %s\n", v ? "int8" : "fp32", err); if (v) arctag_destroy(h[0]); return v ? 77 : 1; }    /* 77 = skipped (no int8 graph) */
    }
    char *txt = slurp(argv[2]); if (!txt) return 1;
    cJSON *ref = cJSON_Parse(txt); free(txt);
    static char ka[2048][200], kb[2048][200]; static int ga[2048], gb[2048]; static double pa[2048], pb[2048];
    int ncase = 0;
    for (cJSON *c = cJSON_GetObjectItem(ref, "cases")->child; c; c = c->next) {
        if (cJSON_GetObjectItem(c, "config")) continue;
        const char *dom = cJSON_GetStringValue(cJSON_GetObjectItem(c, "domain")), *t = cJSON_GetStringValue(cJSON_GetObjectItem(c, "text"));
        cJSON *r[2];
        for (int v = 0; v < 2; v++) { arctag_set_option(h[v], "domain", dom); char *s = arctag_tag_text(h[v], t); r[v] = cJSON_Parse(s); arctag_free_str(s); }
        int na = keys(r[0], ka, ga, pa, 2048), nb = keys(r[1], kb, gb, pb, 2048);
        for (int i = 0; i < na; i++) {
            cnt[ga[i]][0]++;
            for (int j = 0; j < nb; j++) if (!strcmp(ka[i], kb[j])) { cnt[ga[i]][2]++; if (fabs(pa[i] - pb[j]) > maxdp) maxdp = fabs(pa[i] - pb[j]); break; }
        }
        for (int j = 0; j < nb; j++) cnt[gb[j]][1]++;
        cJSON_Delete(r[0]); cJSON_Delete(r[1]); ncase++;
    }
    long A = 0, Bc = 0, C = 0; int bad = 0;
    printf("int8 vs fp32 on %d reference cases (F1 of exact items; groups with fp32 items)\n", ncase);
    for (int g = 0; g < NG; g++) {
        if (!cnt[g][0] && !cnt[g][1]) continue;
        double f1 = cnt[g][0] + cnt[g][1] ? 2.0 * cnt[g][2] / (cnt[g][0] + cnt[g][1]) : 1.0;
        printf("  %-15s fp32 %4ld  int8 %4ld  both %4ld  F1 %.4f\n", GN[g], cnt[g][0], cnt[g][1], cnt[g][2], f1);
        A += cnt[g][0]; Bc += cnt[g][1]; C += cnt[g][2];
    }
    double f1 = A + Bc ? 2.0 * C / (A + Bc) : 1.0;
    printf("  %-15s fp32 %4ld  int8 %4ld  both %4ld  F1 %.4f   max |dp| of shared items %.4f\n", "all", A, Bc, C, f1, maxdp);
    if (f1 < min_f1) { fprintf(stderr, "FAIL: F1 %.4f < %.2f\n", f1, min_f1); bad = 1; }
    if (maxdp > max_dp) { fprintf(stderr, "FAIL: max |dp| %.4f > %.2f\n", maxdp, max_dp); bad = 1; }
    cJSON_Delete(ref); arctag_destroy(h[0]); arctag_destroy(h[1]);
    return bad;
}
