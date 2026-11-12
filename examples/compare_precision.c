/* compare_precision: fp32 and int8 side by side on the same input - load time, tagging time and where the outputs differ.
 *   compare_precision [-m MODEL_DIR] [--backend native|onnxruntime] [-d DOMAIN] [-t THREADS] [-n REPEATS] [INPUT|-]
 * INPUT: one document per line. Prints per document the spans / events / "on" read-outs that only one variant has, and a
 * summary: load ms, mean tagging ms, span agreement (exact type + range), read-out on/off agreement. */
#include "arctag.h"
#include "cJSON.h"
#include "ex_time.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_line(FILE *f) {
    size_t cap = 4096, n = 0; char *b = (char *)malloc(cap); int c;
    while ((c = fgetc(f)) != EOF && c != '\n') { if (n + 1 >= cap) { cap *= 2; b = (char *)realloc(b, cap); } b[n++] = (char)c; }
    if (c == EOF && n == 0) { free(b); return NULL; }
    if (n && b[n - 1] == '\r') n--;
    b[n] = 0; return b;
}

/* "type start end" keys of the spans / "type word" of events / "start key tag" of the on read-outs */
static int keys_of(cJSON *r, char keys[][160], int cap) {
    int n = 0;
    for (cJSON *s = cJSON_GetObjectItem(r, "spans")->child; s && n < cap; s = s->next)
        snprintf(keys[n++], 160, "span %s [%d,%d)", cJSON_GetObjectItem(s, "type")->valuestring, cJSON_GetObjectItem(s, "start")->valueint, cJSON_GetObjectItem(s, "end")->valueint);
    for (cJSON *e = cJSON_GetObjectItem(r, "events")->child; e && n < cap; e = e->next)
        snprintf(keys[n++], 160, "event %s @%d", cJSON_GetObjectItem(e, "type")->valuestring, cJSON_GetObjectItem(e, "word")->valueint);
    for (cJSON *u = cJSON_GetObjectItem(r, "utterances")->child; u; u = u->next)
        for (cJSON *k = u->child; k; k = k->next) {
            if (!cJSON_IsArray(k)) continue;
            for (cJSON *x = k->child; x && n < cap; x = x->next)
                if (cJSON_IsTrue(cJSON_GetObjectItem(x, "on")))
                    snprintf(keys[n++], 160, "utt@%d %s %s", cJSON_GetObjectItem(u, "start")->valueint, k->string, cJSON_GetObjectItem(x, "tag")->valuestring);
        }
    return n;
}

static int has(char keys[][160], int n, const char *k) { for (int i = 0; i < n; i++) if (!strcmp(keys[i], k)) return 1; return 0; }

int main(int argc, char **argv) {
    const char *model = "model", *domain = NULL, *in = "-", *backend = NULL; int threads = 1, reps = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) domain = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "usage: %s [-m MODEL_DIR] [--backend native|onnxruntime] [-d DOMAIN] [-t THREADS] [-n REPEATS] [INPUT|-]\n", argv[0]); return 2; }
        else in = argv[i];
    }
    arctag_t *h[2]; double load[2]; char err[512];
    for (int v = 0; v < 2; v++) {
        arctag_options o; arctag_options_init(&o); o.precision = v ? ARCTAG_INT8 : ARCTAG_FP32; o.threads = threads;
        if (backend) o.backend = !strcmp(backend, "onnxruntime") || !strcmp(backend, "ort") ? ARCTAG_BACKEND_ONNXRUNTIME : ARCTAG_BACKEND_NATIVE;
        double t0 = now_ms(); h[v] = arctag_create_ex(model, &o, err, sizeof err); load[v] = now_ms() - t0;
        if (!h[v]) { fprintf(stderr, "%s: %s\n", v ? "int8" : "fp32", err); if (v) arctag_destroy(h[0]); return v && strstr(err, "arctag_int8.onnx") ? 77 : 1; }   /* 77: no int8 graph in the model dir */
        if (domain && arctag_set_option(h[v], "domain", domain)) { fprintf(stderr, "%s\n", arctag_last_error(h[v])); return 1; }
    }
    FILE *f = strcmp(in, "-") ? fopen(in, "rb") : stdin;
    if (!f) { fprintf(stderr, "cannot open %s\n", in); return 1; }
    static char ka[4096][160], kb[4096][160];
    double ms[2] = {0, 0}; long docs = 0, words = 0, same = 0, tot_a = 0, tot_b = 0; char *line;
    while ((line = read_line(f)) != NULL) {
        if (!*line) { free(line); continue; }
        cJSON *r[2];
        for (int v = 0; v < 2; v++) {
            char *out = NULL;
            for (int k = 0; k < reps; k++) { double t0 = now_ms(); arctag_free_str(out); out = arctag_tag_text(h[v], line); ms[v] += (now_ms() - t0) / reps; }
            r[v] = cJSON_Parse(out); arctag_free_str(out);
        }
        int na = keys_of(r[0], ka, 4096), nb = keys_of(r[1], kb, 4096);
        printf("doc %ld: %d words, fp32 %d items, int8 %d items\n", docs, cJSON_GetObjectItem(r[0], "n_words")->valueint, na, nb);
        for (int i = 0; i < na; i++) if (!has(kb, nb, ka[i])) printf("  fp32 only: %s\n", ka[i]); else same++;
        for (int i = 0; i < nb; i++) if (!has(ka, na, kb[i])) printf("  int8 only: %s\n", kb[i]);
        tot_a += na; tot_b += nb; words += cJSON_GetObjectItem(r[0], "n_words")->valueint; docs++;
        cJSON_Delete(r[0]); cJSON_Delete(r[1]); free(line);
    }
    if (f != stdin) fclose(f);
    printf("summary: %ld documents, %ld words, threads %d\n", docs, words, threads);
    printf("  load ms     fp32 %.1f   int8 %.1f\n", load[0], load[1]);
    printf("  tag ms/word fp32 %.3f   int8 %.3f\n", words ? ms[0] / words : 0, words ? ms[1] / words : 0);
    printf("  agreement   %ld of %ld fp32 items also in int8 (F1 %.4f)\n", same, tot_a, tot_a + tot_b ? 2.0 * same / (tot_a + tot_b) : 1.0);
    arctag_destroy(h[0]); arctag_destroy(h[1]);
    return 0;
}
