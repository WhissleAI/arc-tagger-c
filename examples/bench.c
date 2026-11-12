/* bench: load time, streaming per-word latency, batch throughput and peak RSS of one precision (one process per variant, so
 * the RSS is that variant's).
 *   bench [-m MODEL_DIR] [--int8] [-t THREADS] [-d DOMAIN] [-w WORDS] [-r REPEATS] [--stream-ctx N] FILE...
 * Streaming: the words of the files are pushed one at a time (endpoint after . ? !) and arctag_poll is called after every
 * word; the latency of push+poll is recorded per word (first 32 warm-up words dropped), -w words in total (default 2000).
 * Batch: every line of the files is tagged with arctag_tag_text, -r times (default 3); ms per word = total / words.
 * Prints one JSON line. */
#include "arctag.h"
#include "ex_time.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double pct(double *v, int n, double p) { if (!n) return 0; int k = (int)(p * (n - 1) + 0.5); return v[k]; }
static double peak_rss_mb(void) {
#ifdef _WIN32
    return 0;
#else
    struct rusage u; getrusage(RUSAGE_SELF, &u);
#ifdef __APPLE__
    return u.ru_maxrss / 1048576.0;      /* bytes */
#else
    return u.ru_maxrss / 1024.0;         /* KiB */
#endif
#endif
}

int main(int argc, char **argv) {
    const char *model = getenv("ARCTAG_MODEL_DIR"), *domain = NULL, *sctx = NULL; int int8 = 0, threads = 1, nw = 2000, reps = 3;
    const char *files[64]; int nf = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--int8")) int8 = 1;
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) domain = argv[++i];
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) nw = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stream-ctx") && i + 1 < argc) sctx = argv[++i];
        else if (argv[i][0] == '-') { fprintf(stderr, "usage: %s [-m MODEL_DIR] [--int8] [-t THREADS] [-d DOMAIN] [-w WORDS] [-r REPEATS] [--stream-ctx N] FILE...\n", argv[0]); return 2; }
        else if (nf < 64) files[nf++] = argv[i];
    }
    if (!nf) { fprintf(stderr, "no input files\n"); return 2; }
    /* read lines */
    char **lines = NULL; int nl = 0, cap = 0; long total_words = 0;
    for (int f = 0; f < nf; f++) {
        FILE *fp = fopen(files[f], "rb"); if (!fp) { fprintf(stderr, "cannot open %s\n", files[f]); return 1; }
        char buf[1 << 16];
        while (fgets(buf, sizeof buf, fp)) {
            size_t L = strlen(buf); while (L && (buf[L - 1] == '\n' || buf[L - 1] == '\r')) buf[--L] = 0;
            if (!L) continue;
            if (nl == cap) { cap = cap ? cap * 2 : 64; lines = (char **)realloc(lines, cap * sizeof *lines); }
            lines[nl++] = strdup(buf);
            int in = 0; for (char *p = buf; *p; p++) { int sp = *p == ' ' || *p == '\t'; if (!sp && !in) total_words++; in = !sp; }
        }
        fclose(fp);
    }
    char err[512]; arctag_options o; arctag_options_init(&o); o.precision = int8 ? ARCTAG_INT8 : ARCTAG_FP32; o.threads = threads;
    double t0 = now_ms(); arctag_t *h = arctag_create_ex(model ? model : "model", &o, err, sizeof err); double load = now_ms() - t0;
    if (!h) { fprintf(stderr, "%s\n", err); return int8 && strstr(err, "arctag_int8.onnx") ? 77 : 1; }
    if (domain && arctag_set_option(h, "domain", domain)) { fprintf(stderr, "%s\n", arctag_last_error(h)); return 1; }
    if (sctx && arctag_set_option(h, "stream_ctx", sctx)) { fprintf(stderr, "%s\n", arctag_last_error(h)); return 1; }
    /* streaming */
    double *lat = (double *)malloc(sizeof(double) * (size_t)(nw + 64)); int nlat = 0, pushed = 0;
    char msg[2048];
    while (pushed < nw + 32) {
        for (int l = 0; l < nl && pushed < nw + 32; l++) {
            char *copy = strdup(lines[l]);
            for (char *w = strtok(copy, " \t"); w && pushed < nw + 32; w = strtok(NULL, " \t")) {
                size_t L = strlen(w); int eou = strchr(".?!", w[L - 1]) != NULL;
                char esc[1024]; size_t k = 0;
                for (char *c = w; *c && k + 3 < sizeof esc; c++) { if (*c == '"' || *c == '\\') esc[k++] = '\\'; esc[k++] = *c; }
                esc[k] = 0;
                snprintf(msg, sizeof msg, "{\"words\":[\"%s\"],\"endpoint\":%s}", esc, eou ? "true" : "false");
                double a = now_ms();
                if (arctag_push_words(h, msg)) { fprintf(stderr, "push: %s\n", arctag_last_error(h)); return 1; }
                char *out = arctag_poll(h); arctag_free_str(out);
                double d = now_ms() - a;
                if (pushed++ >= 32) lat[nlat++] = d;
            }
            free(copy);
        }
    }
    qsort(lat, (size_t)nlat, sizeof *lat, cmpd);
    double mean = 0; for (int i = 0; i < nlat; i++) mean += lat[i]; mean /= nlat ? nlat : 1;
    /* batch */
    arctag_reset(h);
    double tb = now_ms();
    for (int r = 0; r < reps; r++)
        for (int l = 0; l < nl; l++) { char *out = arctag_tag_text(h, lines[l]); arctag_free_str(out); }
    double batch_ms_word = (now_ms() - tb) / (double)(total_words * reps);
    char *info = arctag_info(h);
    printf("{\"precision\":\"%s\",\"threads\":%d,\"load_ms\":%.1f,\"stream_words\":%d,\"stream_ms_per_word\":{\"p50\":%.3f,\"p95\":%.3f,\"mean\":%.3f},"
           "\"batch_words\":%ld,\"batch_ms_per_word\":%.4f,\"peak_rss_mb\":%.1f}\n",
           int8 ? "int8" : "fp32", threads, load, nlat, pct(lat, nlat, .5), pct(lat, nlat, .95), mean, total_words * reps, batch_ms_word, peak_rss_mb());
    arctag_free_str(info); free(lat);
    for (int l = 0; l < nl; l++) free(lines[l]);
    free(lines); arctag_destroy(h);
    return 0;
}
