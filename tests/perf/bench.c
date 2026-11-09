/* Speed and memory of the public API, for the native library and the ONNX Runtime one alike (this file only uses arctag.h).
 * usage: bench MODEL_DIR load|text|mem N|stream MAXWORDS. One measurement per process, so memory numbers are clean. */
#include "arctag.h"
#include "mem.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static const char *SENT = "Can we move the budget review to Thursday afternoon and then send Priya the final numbers before noon so the team has time to look them over during the week ahead. ";
static char *words(int n) {
    size_t L = strlen(SENT); char *b = malloc((size_t)n * 16 + L + 1); b[0] = 0; size_t k = 0; int have = 0;
    while (have < n) {
        const char *p = SENT;
        while (*p && have < n) { const char *e = strchr(p, ' '); size_t m = (size_t)(e - p) + 1; memcpy(b + k, p, m); k += m; p = e + 1; have++; }
    }
    b[k] = 0; return b;
}
int main(int c, char **v) {
    if (c < 3) { fprintf(stderr, "usage: %s MODEL_DIR load|text|mem N|stream MAXWORDS\n", v[0]); return 2; }
    char err[512]; double r0, d0; mem_now(&r0, &d0);
    double t = now_ms(); arctag_t *h = arctag_create(v[1], err, sizeof err); double load = now_ms() - t;
    if (!h) { fprintf(stderr, "%s\n", err); return 1; }
    arctag_set_option(h, "threads", "1");
    if (!strcmp(v[2], "load")) {
        t = now_ms(); char *o = arctag_tag_text(h, "Can you call me back tomorrow?"); double first = now_ms() - t; arctag_free_str(o);
        printf("load %.1f ms, first short text %.1f ms\n", load, first);
    } else if (!strcmp(v[2], "text")) {
        int ns[] = {8, 32, 150, 512};
        for (int k = 0; k < 4; k++) {
            char *b = words(ns[k]); int runs = ns[k] > 200 ? 15 : 40; double *d = malloc(sizeof(double) * (size_t)runs);
            for (int i = 0; i < runs; i++) { t = now_ms(); char *o = arctag_tag_text(h, b); d[i] = now_ms() - t; arctag_free_str(o); }
            qsort(d, (size_t)runs, sizeof(double), cmpd);
            printf("%d words: min %.2f median %.2f p90 %.2f ms\n", ns[k], d[0], d[runs / 2], d[runs * 9 / 10]);
            free(d); free(b);
        }
    } else if (!strcmp(v[2], "mem")) {
        int n = c > 3 ? atoi(v[3]) : 150; char *b = words(n);
        for (int i = 0; i < 5; i++) { char *o = arctag_tag_text(h, b); arctag_free_str(o); }
        double r1, d1; mem_now(&r1, &d1);
        printf("%d words: added rss %.1f MB, added dirty %.1f MB (rss %.1f, dirty %.1f)\n", n, r1 - r0, d1 - d0, r1, d1);
        free(b);
    } else if (!strcmp(v[2], "stream")) {
        int maxw = c > 3 ? atoi(v[3]) : 2047; char *b = words(maxw + 4); char **w = malloc(sizeof(char *) * (size_t)(maxw + 4)); int nw = 0;
        for (char *x = strtok(b, " "); x; x = strtok(NULL, " ")) w[nw++] = x;
        double *us = malloc(sizeof(double) * (size_t)maxw); char js[256]; int marks[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2047};
        for (int i = 0; i < maxw; i++) {
            snprintf(js, sizeof js, "[\"%s\"]", w[i]);
            t = now_ms(); arctag_push_words(h, js); char *o = arctag_poll(h); us[i] = now_ms() - t; arctag_free_str(o);
            for (int m = 0; m < 9; m++) if (marks[m] == i + 1) {
                int s = i + 1 >= 16 ? i + 1 - 16 : 0, n = i + 1 - s; double tmp[16]; memcpy(tmp, us + s, sizeof(double) * (size_t)n); qsort(tmp, (size_t)n, sizeof(double), cmpd);
                double r1, d1; mem_now(&r1, &d1);
                printf("context %5d: median per new word %.2f ms; rss %.1f MB dirty %.1f MB\n", i + 1, tmp[n / 2], r1, d1);
            }
        }
        double ts[20];
        for (int r = 0; r < 20; r++) {
            snprintf(js, sizeof js, "{\"words\":[\"%s\",\"%s\",\"%s\"],\"from\":%d}", w[maxw - 3], w[maxw - 2], w[maxw - 1], maxw - 3);
            t = now_ms(); arctag_push_words(h, js); char *o = arctag_poll(h); ts[r] = now_ms() - t; arctag_free_str(o);
        }
        qsort(ts, 20, sizeof(double), cmpd);
        printf("revise last 3 words + poll at %d: median %.2f ms\n", maxw, ts[10]);
    }
    arctag_destroy(h);
    return 0;
}
